"""
Standard Energy Index (SEI) computation functions.

This module provides functions for computing the Standard Energy Index (SEI)
using an Empirical Cumulative Distribution Function (ECDF) approach with
seasonal awareness (day-of-year basis).

SEI is used to identify drought periods in renewable generation or high/low
demand periods in energy demand data.

Reference: https://www.sciencedirect.com/science/article/pii/S0960148123011217?via%3Dihub#d1e894
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF


def unstack_time_to_doy_year(data):
    """
    Convert a timeseries from 'time' dimension to (dayofyear, year) dimensions.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data with a 'time' dimension containing datetime coordinates.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Data restructured with 'dayofyear' and 'year' dimensions instead of 'time'.

    Examples
    --------
    >>> data_doy = unstack_time_to_doy_year(timeseries_data)
    >>> print(data_doy.dims)  # ('dayofyear', 'year', ...)
    """
    result = data.copy(deep=True)

    # Add day-of-year and year coordinates
    result["dayofyear"] = result.time.dt.dayofyear
    result["year"] = result.time.dt.year
    result = result.assign_coords(
        {"dayofyear": result.time.dt.dayofyear, "year": result.time.dt.year}
    )

    # Reshape: separate time into (dayofyear, year)
    result = result.drop_vars("time").set_index(time=["dayofyear", "year"]).unstack()

    return result


def stack_doy_year_to_time(data, calendar="noleap", start_year=None):
    """
    Convert data from (dayofyear, year) dimensions back to a single 'time' dimension.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data with 'dayofyear' and 'year' dimensions.
    calendar : str, optional
        Calendar type for the time coordinate. Default is 'noleap'.
        Options: 'standard', 'gregorian', 'noleap', '365_day', '360_day', etc.
    start_year : int, optional
        If provided, filters data to start from this year. Useful for handling
        filled NaN years at the beginning of the dataset.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Data restructured with a single 'time' dimension.

    Examples
    --------
    >>> timeseries = stack_doy_year_to_time(data_doy)
    >>> print(timeseries.dims)  # ('time', ...)

    >>> # Filter to start from specific year
    >>> timeseries = stack_doy_year_to_time(data_doy, start_year=1980)
    """
    result = data.copy(deep=True)

    # Filter to start year if provided
    if start_year is not None:
        result = result.sel(year=slice(start_year, None))

    # Stack year and dayofyear back into a single dimension
    result = result.stack(time=["year", "dayofyear"])

    # Create proper datetime coordinates
    # The stacked coordinate is a MultiIndex with (year, dayofyear) tuples
    time_index = result.time.to_index()

    # Construct datetime64 array
    time_coords = []
    for year, doy in time_index:
        # Create date from year and day-of-year
        if calendar in ["noleap", "365_day"]:
            # For noleap calendar, directly use dayofyear
            date = pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(
                days=int(doy) - 1
            )
        else:
            # For other calendars, use standard datetime
            date = pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(
                days=int(doy) - 1
            )
        time_coords.append(date)

    # Reset the time coordinate to be a simple datetime coordinate
    result = result.reset_index("time", drop=True)
    result = result.assign_coords(time=("time", time_coords))

    # Convert to specified calendar if not standard
    if calendar != "standard" and calendar != "gregorian":
        result = result.convert_calendar(calendar)

    return result


def compute_standardized_index(ds_window_slice, ds_doy_slice):
    """
    Computes the standardized index for a single day-of-year slice.
    Applied by xarray.apply_ufunc across all days of year.

    Parameters
    ----------
    ds_window_slice : numpy.ndarray
        The rolling window slice of historical data for ECDF fitting (flattened).
    ds_doy_slice : numpy.ndarray
        The single day-of-year data point to compute the index for.

    Returns
    -------
    float
        Standardized index (Z-score), or NaN if target value is NaN or insufficient data
    """
    # If target value is NaN, return NaN immediately
    if np.isnan(ds_doy_slice):
        return np.nan

    # Fit ECDF to the reference window samples, excluding NaN values
    historical_samples = ds_window_slice.flatten()
    historical_samples = historical_samples[~np.isnan(historical_samples)]

    # If no valid samples, return NaN
    if len(historical_samples) == 0:
        return np.nan

    ecdf_func = ECDF(historical_samples)

    # Get the raw probability F_n(x) for the day-of-year value
    fn_x = ecdf_func(ds_doy_slice)

    # Apply ECDF smoothing/rescaling: (n * F_n(x) + 1) / (n + 2)
    # This prevents exactly 0 or 1 probabilities, keeping Z-scores finite
    n = len(historical_samples)
    f_x_rescaled = (n * fn_x + 1) / (n + 2)

    # Convert probability to Z-score (standardized index)
    standardized_index = norm.ppf(f_x_rescaled)

    return standardized_index


def compute_sei(reference_data, target_data, window_size=60, fill_missing_year=True):
    """
    Compute Standard Energy Index (SEI) for a timeseries using ECDF approach.

    SEI is a seasonally-aware standardized index (Z-score) where:
    - Negative values indicate below-normal conditions
    - Positive values indicate above-normal conditions
    - Values around 0 indicate near-normal conditions

    Parameters
    ----------
    reference_data : xr.DataArray
        Reference period data used to build the ECDF for each day-of-year.
        Must have a 'time' dimension with datetime coordinates.
        Typically a historical period (e.g., 1982-2011).
    target_data : xr.DataArray
        Target period data for which to compute SEI values.
        Must have a 'time' dimension with datetime coordinates.
        Can be the same as reference_data or a different period (e.g., future projections).
    window_size : int, optional
        Number of days for rolling window centered on each day-of-year.
        Default is 60 (±30 days around each DOY).
        Larger windows = more samples but less seasonal specificity.
    fill_missing_year : bool, optional
        If True, automatically detects and fills missing years (e.g., 2014 for WRF data)
        with NaN values after SEI computation. Default is True.
        This is useful for WRF simulations which have a gap between historical and SSP scenarios.

    Returns
    -------
    xr.DataArray
        SEI values with dimensions matching target_data structure,
        with 'time' replaced by 'year' and 'dayofyear' dimensions.
        If fill_missing_year=True, any missing years in the sequence will be filled with NaN.

    Examples
    --------
    >>> # For renewable energy drought analysis
    >>> sei = compute_sei(historical_gen, future_gen, window_size=30)
    >>>
    >>> # For demand analysis
    >>> sei = compute_sei(historical_demand, future_demand, window_size=60)
    >>>
    >>> # If your data doesn't have missing years, disable auto-fill
    >>> sei = compute_sei(reference, target, fill_missing_year=False)

    Notes
    -----
    - Both inputs should use 'noleap' calendar for consistent day-of-year mapping
    - The function handles missing data (NaN values) appropriately
    - WRF simulations have a gap in 2014 between historical and SSP scenarios.
      If fill_missing_year=True, this gap will be filled with NaN values automatically.
    - For demand: positive SEI = high demand, negative = low demand
    - For generation: positive SEI = high generation, negative = low generation (drought)
    """

    # Step 1: Prepare reference data with rolling window for ECDF fitting
    # Create rolling window centered on each day
    ds_window = reference_data.rolling(time=window_size, center=True).construct(
        "window"
    )
    ds_window = ds_window.assign_coords({"window": np.arange(1, window_size + 1)})

    # Add day-of-year and year coordinates
    ds_window["dayofyear"] = ds_window.time.dt.dayofyear
    ds_window["year"] = ds_window.time.dt.year
    ds_window = ds_window.assign_coords(
        {"dayofyear": ds_window.time.dt.dayofyear, "year": ds_window.time.dt.year}
    )

    # Reshape: separate time into (dayofyear, year) dimensions
    ds_window = (
        ds_window.drop_vars("time").set_index(time=["dayofyear", "year"]).unstack()
    )

    # Flatten window and year dimensions into single 'sample' dimension
    # This gives us all samples for each day-of-year to build the ECDF
    ds_window = ds_window.stack(sample=["window", "year"])

    # Step 2: Prepare target data in day-of-year structure
    ds_doy = target_data.copy(deep=True)
    ds_doy["dayofyear"] = ds_doy.time.dt.dayofyear
    ds_doy["year"] = ds_doy.time.dt.year
    ds_doy = ds_doy.assign_coords(
        {"dayofyear": ds_doy.time.dt.dayofyear, "year": ds_doy.time.dt.year}
    )

    # Reshape: separate time into (dayofyear, year)
    ds_doy = ds_doy.drop_vars("time").set_index(time=["dayofyear", "year"]).unstack()

    # Step 3: Apply SEI computation
    # For each day-of-year and year in target data, compute SEI using the
    # corresponding day-of-year ECDF from reference data
    standardized_index = xr.apply_ufunc(
        compute_standardized_index,
        ds_window,
        ds_doy,
        input_core_dims=[["sample"], []],  # 'sample' for ECDF, single value for target
        output_core_dims=[[]],
        vectorize=True,  # Apply over all other dimensions
        dask="parallelized",  # Support dask arrays
        output_dtypes=[np.float32],
        exclude_dims=set(("sample",)),  # 'sample' dimension consumed in function
    )

    standardized_index.name = "sei"
    standardized_index.attrs["long_name"] = "Standard Energy Index"
    standardized_index.attrs["description"] = (
        "Seasonally-aware standardized index (Z-score) relative to reference period"
    )
    standardized_index.attrs["window_size"] = window_size

    # Step 4: Fill missing years if requested
    if fill_missing_year:
        # Check if there are missing years in the sequence
        years = standardized_index.year.values
        year_min, year_max = int(years.min()), int(years.max())
        expected_years = np.arange(year_min, year_max + 1)
        missing_years = set(expected_years) - set(years)

        if missing_years:
            # Create NaN-filled data for missing years
            missing_datasets = []
            for missing_year in sorted(missing_years):
                # Get spatial dimensions from the original data (if any)
                other_dims = {
                    dim: standardized_index[dim]
                    for dim in standardized_index.dims
                    if dim not in ["year", "dayofyear"]
                }

                # Create NaN array with proper shape
                shape = [365] + [
                    len(standardized_index[dim]) for dim in other_dims.keys()
                ]
                nan_data = np.full(shape, np.nan, dtype=np.float32)

                # Build coordinates
                coords = {"dayofyear": np.arange(1, 366), "year": missing_year}
                coords.update(other_dims)

                # Build dimensions list
                dims = ["dayofyear"] + list(other_dims.keys())

                # Create DataArray for this missing year
                missing_da = xr.DataArray(
                    nan_data, coords=coords, dims=dims, name="sei"
                )
                missing_datasets.append(missing_da)

            # Concatenate with original data and sort by year
            standardized_index = xr.concat(
                [standardized_index] + missing_datasets, dim="year"
            ).sortby("year")

            # Restore attributes
            standardized_index.name = "sei"
            standardized_index.attrs["long_name"] = "Standard Energy Index"
            standardized_index.attrs["description"] = (
                "Seasonally-aware standardized index (Z-score) relative to reference period"
            )
            standardized_index.attrs["window_size"] = window_size
            standardized_index.attrs["filled_years"] = sorted(list(missing_years))

    return standardized_index
