import argparse
import time

import numpy as np
import xarray as xr
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF

from src.renewable_data_load import *


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate daily Standardized Energy Index (SEI) for renewable energy data."
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
        default="miroc6",
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
        "--output-dir",
        type=str,
        default="../data/SEI",
        help="Output directory for zarr files",
    )

    return parser.parse_args()


def main():
    start_time = time.time()
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

    print("=" * 60)
    print("SEI Calculation Parameters:")
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
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    ren_ds = get_ren_data_concat(
        resource, module, domain, variable, frequency, simulation
    )

    #### SUBSET FOR TESTING ####
    # Uncomment the line below for testing with a small spatial subset
    ren_ds = ren_ds.isel(y=slice(222, 226), x=slice(53, 57))

    print(f"Data shape: {ren_ds.sizes}")
    print(f"Memory footprint: {ren_ds.nbytes / 1e9:.2f} GB")

    # Get bounds for reference GWL period
    print("\nDetermining reference period from GWL...")
    WRF_sim_name = sim_name_dict[simulation]
    model = WRF_sim_name.split("_")[1]
    ensemble_member = WRF_sim_name.split("_")[2]
    start_year, end_year = get_gwl_crossing_period(
        model, ensemble_member, reference_gwl
    )
    print(f"Reference period (GWL {reference_gwl}°C): {start_year}-{end_year}")

    # Extract reference periods
    print("\nExtracting reference period...")
    ds_reference = ren_ds.sel(time=slice(f"{start_year}", f"{end_year}"))

    ## Reshape Future Data to (dayofyear, year, y, x)
    print("Reshaping data to (dayofyear, year, y, x)...")
    ds_doy = ren_ds.copy(deep=True)

    # Add temporal coordinates
    ds_doy["dayofyear"] = ds_doy.time.dt.dayofyear
    ds_doy["year"] = ds_doy.time.dt.year

    # Reshape: time -> (dayofyear, year)
    ds_doy = ds_doy.drop_vars("time").set_index(time=["dayofyear", "year"]).unstack()

    # Remove 2014 if present (GCM artifact)
    if simulation != "ERA5" and 2014 in ds_doy.year:
        ds_doy = ds_doy.drop_sel(year=2014)

    print(f"Reshaped data: {ds_doy.sizes}")

    ## Compute SEI for Each Day-of-Year
    print("\nComputing SEI for each day-of-year...")
    print("This may take several minutes...")

    # List to collect SEI for each day-of-year
    sei_list = []
    gen_list = []

    # Process each day-of-year
    for doy in range(1, 366):
        if doy % 50 == 0:
            print(f"Processing day {doy}/365...")

        # Get future generation data for this DOY
        future_doy_data = ds_doy.sel(dayofyear=doy)

        # Compute SEI for this DOY using parallelized apply_ufunc
        sei_doy = compute_sei_for_doy_parallel(
            ds_reference, future_doy_data, doy, window_size
        )

        # Add dayofyear coordinate
        sei_doy = sei_doy.expand_dims(dayofyear=[doy])
        gen_doy = future_doy_data.expand_dims(dayofyear=[doy])

        sei_list.append(sei_doy)
        gen_list.append(gen_doy)

    print("Concatenating results...")

    # Concatenate all days-of-year
    sei_full = xr.concat(sei_list, dim="dayofyear")
    gen_full = xr.concat(gen_list, dim="dayofyear")

    # Merge into final dataset
    drought_ds = xr.Dataset({"gen": gen_full, "sei": sei_full})

    print("Computing final dataset (loading into memory)...")
    drought_ds = drought_ds.compute()

    # Add missing 2014 year if needed
    if simulation != "ERA5":
        print("Adding placeholder for missing year 2014...")
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

    #### SAVE to zarr
    output_path = f"{args.output_dir}/{resource}_{module}_{domain}_{variable}_daily_sei_{simulation}_{scenario}_gwl{reference_gwl}.zarr"

    print("\nPreparing to save...")

    # Remove encoding to avoid zarr version conflicts
    for var in drought_ds.data_vars:
        if "encoding" in drought_ds[var].attrs:
            del drought_ds[var].attrs["encoding"]
        drought_ds[var].encoding = {}

    for coord in drought_ds.coords:
        if "encoding" in drought_ds[coord].attrs:
            del drought_ds[coord].attrs["encoding"]
        drought_ds[coord].encoding = {}

    # Add metadata
    drought_ds.attrs["title"] = f"Standardized Energy Index - {resource} {module}"
    drought_ds.attrs["simulation"] = simulation
    drought_ds.attrs["scenario"] = scenario
    drought_ds.attrs["reference_gwl"] = reference_gwl
    drought_ds.attrs["reference_period"] = f"{start_year}-{end_year}"
    drought_ds.attrs["window_size_days"] = window_size
    drought_ds.attrs["method"] = (
        "memory-efficient dynamic window extraction with apply_ufunc parallelization"
    )

    print(f"Saving to: {output_path}")
    drought_ds.to_zarr(output_path, mode="w")
    print("Save complete!")
    print("=" * 60)
    print("SEI calculation finished successfully!")
    print("=" * 60)
    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
