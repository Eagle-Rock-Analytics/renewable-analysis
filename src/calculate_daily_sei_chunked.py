import argparse
import gc
import os
import time

import numpy as np
import xarray as xr
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF


def get_ren_data_concat(resource, module, domain, variable, frequency, simulation):

    # historical
    path = f"s3://wfclimres/era/{resource}_{module}/{simulation}/historical/{frequency}/{variable}/{domain}/"
    hist_ds = xr.open_zarr(path, storage_options={"anon": True})
    hist_ds = hist_ds.convert_calendar("noleap")
    path = f"s3://wfclimres/era/{resource}_{module}/{simulation}/ssp370/{frequency}/{variable}/{domain}/"
    fut_ds = xr.open_zarr(path, storage_options={"anon": True})
    fut_ds = fut_ds.convert_calendar("noleap")

    # combine historical and future
    ds = xr.concat([hist_ds, fut_ds], dim="time")
    ds = ds.convert_calendar("noleap")

    ds = ds[variable]
    ds = ds.isel(
        x=slice(10, -10), y=slice(10, -10)
    )  # trim the edges to match the WRF AE domain

    # This does not work well as a merge or a concat because of the time and gwl dimensions both being there. looking into this.
    # comb_ds = xr.merge(gwl_list)
    return ds


def extract_window_samples(ds_reference, doy, window_size=60):
    """
    Extract window samples for a single day-of-year from reference period.

    Parameters
    ----------
    ds_reference : xarray.Dataset
        Reference period data with time dimension
    doy : int
        Day-of-year to extract window for (1-365)
    window_size : int
        Window size in days (centered on doy)

    Returns
    -------
    window_data : xarray.DataArray
        Window samples with dimensions (sample, y, x)
    """
    half_window = window_size // 2

    # Handle circular day-of-year for window edges
    doy_min = doy - half_window
    doy_max = doy + half_window

    # Extract time indices where dayofyear falls in window
    reference_doy = ds_reference.time.dt.dayofyear

    if doy_min < 1:
        # Wrap around year boundary (e.g., doy=5, window includes doy=340-365 and 1-35)
        mask = (reference_doy >= (365 + doy_min)) | (reference_doy <= doy_max)
    elif doy_max > 365:
        # Wrap around year boundary (e.g., doy=361, window includes 331-365 and 1-26)
        mask = (reference_doy >= doy_min) | (reference_doy <= (doy_max - 365))
    else:
        # Normal case - no wrap-around
        mask = (reference_doy >= doy_min) & (reference_doy <= doy_max)

    # Extract and flatten time dimension to 'sample'
    window_data = ds_reference.where(mask, drop=True)
    window_data = window_data.stack(sample=["time"])

    return window_data


def compute_sei_vectorized(reference_samples, future_value):
    """
    Vectorized SEI computation for a single spatial point.

    This function is designed to work with xr.apply_ufunc and operates on numpy arrays.

    Parameters
    ----------
    reference_samples : np.ndarray
        1D array of historical samples from the window
    future_value : float or np.ndarray
        Future value(s) to compute SEI for

    Returns
    -------
    sei : float or np.ndarray
        Standardized energy index (Z-score)
    """
    # Remove NaN values
    valid_samples = reference_samples[~np.isnan(reference_samples)]

    if len(valid_samples) == 0:
        # No valid data - return NaN
        return np.full_like(future_value, np.nan, dtype=np.float32)

    # Fit ECDF to historical distribution
    ecdf_func = ECDF(valid_samples)

    # Compute probabilities
    fn_x = ecdf_func(future_value)

    # Apply smoothing: (n * F_n(x) + 1) / (n + 2)
    n = len(valid_samples)
    f_x_rescaled = (n * fn_x + 1) / (n + 2)

    # Convert to Z-scores
    sei = norm.ppf(f_x_rescaled)

    return sei.astype(np.float32)


def compute_sei_for_doy_parallel(ds_reference, ds_doy, doy, window_size=60):
    """
    Compute SEI for a single day-of-year using apply_ufunc for parallelization.

    Parameters
    ----------
    ds_reference : xarray.Dataset
        Reference period data with time dimension for ECDF fitting
    ds_doy : xarray.DataArray
        Future data for this specific day-of-year (year, y, x)
    doy : int
        Day-of-year to process (1-365)
    window_size : int
        Window size in days (centered on doy)

    Returns
    -------
    sei : xarray.DataArray
        Standardized energy index (year, y, x)
    """
    # Extract window samples for this DOY
    window_samples = extract_window_samples(ds_reference, doy, window_size)

    # Use apply_ufunc for vectorized, parallelized computation
    sei = xr.apply_ufunc(
        compute_sei_vectorized,
        window_samples,  # (sample, y, x)
        ds_doy,  # (year, y, x)
        input_core_dims=[
            ["sample"],
            [],
        ],  # 'sample' is core dim for ECDF, none for future values
        output_core_dims=[[]],  # Output has no additional core dims
        vectorize=True,  # Vectorize over (y, x, year)
        dask="parallelized",  # Enable parallel execution with dask
        output_dtypes=[np.float32],  # Output data type
        exclude_dims=set(("sample",)),  # 'sample' dimension consumed in function
        dask_gufunc_kwargs={"allow_rechunk": True},  # Allow rechunking for efficiency
    )

    sei.name = "sei"
    return sei


def process_spatial_chunk(
    ren_ds_chunk,
    ds_reference_chunk,
    simulation,
    window_size,
    chunk_x_idx,
    chunk_y_idx,
):
    """
    Process a single spatial chunk to compute SEI.

    Parameters
    ----------
    ren_ds_chunk : xarray.Dataset
        Spatial chunk of full dataset
    ds_reference_chunk : xarray.Dataset
        Spatial chunk of reference period
    simulation : str
        Simulation name
    window_size : int
        Window size for ECDF
    chunk_x_idx : int
        X chunk index
    chunk_y_idx : int
        Y chunk index

    Returns
    -------
    drought_ds : xarray.Dataset
        SEI results for this chunk
    """
    print(f"  Processing chunk (y={chunk_y_idx}, x={chunk_x_idx})...")

    ## Reshape Future Data to (dayofyear, year, y, x)
    ds_doy = ren_ds_chunk.copy(deep=True)

    # Add temporal coordinates
    ds_doy["dayofyear"] = ds_doy.time.dt.dayofyear
    ds_doy["year"] = ds_doy.time.dt.year

    # Reshape: time -> (dayofyear, year)
    ds_doy = ds_doy.drop_vars("time").set_index(time=["dayofyear", "year"]).unstack()

    # Remove 2014 if present (GCM artifact)
    if simulation != "ERA5" and 2014 in ds_doy.year:
        ds_doy = ds_doy.drop_sel(year=2014)

    ## Compute SEI for Each Day-of-Year
    # List to collect SEI for each day-of-year
    sei_list = []
    gen_list = []

    # Process each day-of-year
    for doy in range(1, 366):
        # Get future generation data for this DOY
        future_doy_data = ds_doy.sel(dayofyear=doy)

        # Compute SEI for this DOY using parallelized apply_ufunc
        sei_doy = compute_sei_for_doy_parallel(
            ds_reference_chunk, future_doy_data, doy, window_size
        )

        # Add dayofyear coordinate
        sei_doy = sei_doy.expand_dims(dayofyear=[doy])
        gen_doy = future_doy_data.expand_dims(dayofyear=[doy])

        sei_list.append(sei_doy)
        gen_list.append(gen_doy)

    # Concatenate all days-of-year
    sei_full = xr.concat(sei_list, dim="dayofyear")
    gen_full = xr.concat(gen_list, dim="dayofyear")

    # Merge into final dataset
    drought_ds = xr.Dataset({"gen": gen_full, "sei": sei_full})

    # Compute to materialize results
    drought_ds = drought_ds.compute()

    # Add missing 2014 year if needed
    if simulation != "ERA5":
        synth = np.zeros(shape=[365, 1])
        synth[:, :] = np.nan
        fill_data = xr.Dataset(
            data_vars={
                "gen": (["dayofyear", "year"], synth),
                "sei": (["dayofyear", "year"], synth),
            },
            coords={"year": [2014], "dayofyear": np.arange(1, 366, 1)},
        )
        drought_ds = xr.merge([fill_data, drought_ds])

    # Add chunk indices as attributes
    drought_ds.attrs["chunk_x_idx"] = chunk_x_idx
    drought_ds.attrs["chunk_y_idx"] = chunk_y_idx

    return drought_ds


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate daily Standardized Energy Index (SEI) for renewable energy data in spatial chunks."
    )

    parser.add_argument(
        "--resource", type=str, default="pv", help="Resource type (e.g., pv, windpower)"
    )
    parser.add_argument(
        "--module",
        type=str,
        default="distributed",
        help="Module type (e.g., distributed, utility, onshore, offshore)",
    )
    parser.add_argument(
        "--domain", type=str, default="d03", help="Domain (e.g., d02, d03)"
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="gen",
        help="Variable to analyze (e.g., gen, cf)",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="day",
        help="Temporal frequency (e.g., day, 1hr)",
    )
    parser.add_argument(
        "--simulation",
        type=str,
        default="ec-earth3",
        help="Climate simulation (e.g., ec-earth3, miroc6, mpi-esm1-2-hr)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="historical",
        help="Climate scenario (e.g., historical, ssp370)",
    )
    parser.add_argument(
        "--reference-gwl",
        type=float,
        default=0.8,
        help="Reference global warming level (e.g., 0.8, 1.5, 2.0)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Window size in days for ECDF fitting (default: 60)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Spatial chunk size for both x and y dimensions (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/SEI",
        help="Output directory for NetCDF files",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_args()

    resource = args.resource
    module = args.module
    domain = args.domain
    variable = args.variable
    frequency = args.frequency
    simulation = args.simulation
    scenario = args.scenario
    reference_gwl = args.reference_gwl
    window_size = args.window_size
    chunk_size = args.chunk_size

    print("=" * 60)
    print("SEI Calculation Parameters (Chunked):")
    print("=" * 60)
    print(f"Resource: {resource}")
    print(f"Module: {module}")
    print(f"Domain: {domain}")
    print(f"Variable: {variable}")
    print(f"Frequency: {frequency}")
    print(f"Simulation: {simulation}")
    print(f"Scenario: {scenario}")
    print(f"Reference GWL: {reference_gwl}°C")
    print(f"Window size: {window_size} days")
    print(f"Spatial chunk size: {chunk_size}x{chunk_size}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    ren_ds = get_ren_data_concat(
        resource, module, domain, variable, frequency, simulation
    )

    print(f"Full data shape: {ren_ds.sizes}")
    print(f"Total memory footprint: {ren_ds.nbytes / 1e9:.2f} GB")

    # Get bounds for reference GWL period
    print("\nDetermining reference period from GWL...")
    WRF_sim_name = sim_name_dict[simulation]
    model = WRF_sim_name.split("_")[1]
    ensemble_member = WRF_sim_name.split("_")[2]
    start_year, end_year = get_gwl_crossing_period(
        model, ensemble_member, reference_gwl
    )
    print(f"Reference period (GWL {reference_gwl}°C): {start_year}-{end_year}")

    # Extract reference period (full spatial domain)
    print("\nExtracting reference period...")
    ds_reference = ren_ds.sel(time=slice(f"{start_year}", f"{end_year}"))

    # Determine chunk boundaries
    ny = ren_ds.sizes["y"]
    nx = ren_ds.sizes["x"]

    y_chunks = list(range(0, ny, chunk_size))
    x_chunks = list(range(0, nx, chunk_size))

    n_y_chunks = len(y_chunks)
    n_x_chunks = len(x_chunks)

    print(f"\nSpatial dimensions: y={ny}, x={nx}")
    print(
        f"Number of chunks: {n_y_chunks} (y) × {n_x_chunks} (x) = {n_y_chunks * n_x_chunks} total"
    )

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each spatial chunk
    chunk_count = 0
    total_chunks = n_y_chunks * n_x_chunks

    for y_idx, y_start in enumerate(y_chunks):
        y_end = min(y_start + chunk_size, ny)

        for x_idx, x_start in enumerate(x_chunks):
            x_end = min(x_start + chunk_size, nx)

            chunk_count += 1

            # Prepare output path with chunk indices
            output_filename = (
                f"{resource}_{module}_{domain}_{variable}_daily_sei_"
                f"{simulation}_{scenario}_gwl{reference_gwl}_"
                f"chunk_y{y_idx:03d}_x{x_idx:03d}.nc"
            )
            output_path = os.path.join(args.output_dir, output_filename)

            # Check if chunk already exists
            if os.path.exists(output_path):
                print(f"\n{'=' * 60}")
                print(
                    f"Chunk {chunk_count}/{total_chunks}: y[{y_start}:{y_end}], x[{x_start}:{x_end}]"
                )
                print(f"SKIPPING - Already exists: {output_filename}")
                print(f"{'=' * 60}")
                continue

            chunk_start_time = time.time()

            print(f"\n{'=' * 60}")
            print(
                f"Chunk {chunk_count}/{total_chunks}: y[{y_start}:{y_end}], x[{x_start}:{x_end}]"
            )
            print(f"{'=' * 60}")

            # Extract spatial chunk
            ren_ds_chunk = ren_ds.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
            ds_reference_chunk = ds_reference.isel(
                y=slice(y_start, y_end), x=slice(x_start, x_end)
            )

            print(f"Chunk shape: {ren_ds_chunk.sizes}")
            print(f"Chunk memory: {ren_ds_chunk.nbytes / 1e9:.2f} GB")

            # Process chunk
            drought_ds_chunk = process_spatial_chunk(
                ren_ds_chunk,
                ds_reference_chunk,
                simulation,
                window_size,
                x_idx,
                y_idx,
            )

            # Add global metadata
            drought_ds_chunk.attrs["title"] = (
                f"Standardized Energy Index - {resource} {module}"
            )
            drought_ds_chunk.attrs["simulation"] = simulation
            drought_ds_chunk.attrs["scenario"] = scenario
            drought_ds_chunk.attrs["reference_gwl"] = reference_gwl
            drought_ds_chunk.attrs["reference_period"] = f"{start_year}-{end_year}"
            drought_ds_chunk.attrs["window_size_days"] = window_size
            drought_ds_chunk.attrs["method"] = (
                "memory-efficient dynamic window extraction with apply_ufunc parallelization"
            )
            drought_ds_chunk.attrs["y_start"] = y_start
            drought_ds_chunk.attrs["y_end"] = y_end
            drought_ds_chunk.attrs["x_start"] = x_start
            drought_ds_chunk.attrs["x_end"] = x_end

            # Save chunk as NetCDF
            print(f"Saving chunk to: {output_path}")
            drought_ds_chunk.to_netcdf(output_path)

            chunk_elapsed = time.time() - chunk_start_time
            print(
                f"Chunk {chunk_count}/{total_chunks} completed in {chunk_elapsed:.1f}s"
            )

            # Force cleanup to prevent memory accumulation
            del drought_ds_chunk, ren_ds_chunk, ds_reference_chunk
            gc.collect()
            print(f"Memory cleanup complete")

    print("\n" + "=" * 60)
    print("All chunks processed successfully!")
    print(f"Total chunks saved: {total_chunks}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
