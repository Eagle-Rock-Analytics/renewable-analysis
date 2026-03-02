"""
Climate correlation analysis utilities.

Shared functions for correlating renewable energy metrics, demand, and climate variables.
Supports both gridded, regional, and point-based analysis with flexible anomaly calculations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from climakitae.core.data_interface import get_data
from shapely.geometry import Point, mapping

from renewable_data_load import (
    get_gwl_crossing_period,
    get_ren_data_concat,
    resolution_dict,
    scenario_dict,
    sim_name_dict,
)
from sei_computation import compute_sei


def get_wrf_crs() -> pyproj.CRS:
    """
    Get the WRF Lambert Conformal coordinate reference system.

    Returns
    -------
    pyproj.CRS
        WRF projection CRS
    """
    return pyproj.CRS(
        "+proj=lcc +lat_0=38. +lon_0=-70. +lat_1=30. "
        "+lat_2=60. +R=6370000. +units=m +no_defs"
    )


def load_wrf_climate_data_all_simulations(
    variable: str, domain: str = "d02", trim_edges: bool = True
) -> xr.DataArray:
    """
    Load WRF climate data for all simulations at once (efficient single call).

    This function calls get_data() once and returns data for all simulations.
    Use extract_simulation_gwl_period() to extract specific simulation/GWL periods.

    Parameters
    ----------
    variable : str
        Climate variable name (e.g., 'Air Temperature at 2m', 'Wind Speed at 10m')
    domain : str, optional
        WRF domain ('d02' or 'd03'), default 'd02'
    trim_edges : bool, optional
        Whether to trim edges to match AE domain, default True

    Returns
    -------
    xr.DataArray
        Climate data for all simulations with noleap calendar
    """
    # Set up data retrieval
    wrf_scenario = ["Historical Climate", "SSP 3-7.0"]
    wrf_resolution = resolution_dict[domain]

    # Retrieve data (returns all 8 simulations)
    data = get_data(
        variable=variable,
        downscaling_method="Dynamical",
        resolution=wrf_resolution,
        timescale="daily",
        scenario=wrf_scenario,
    )

    # Convert to noleap calendar
    data = data.convert_calendar("noleap")

    # Trim edges if requested
    if trim_edges:
        data = data.isel(x=slice(10, -10), y=slice(10, -10))

    return data


def extract_simulation_gwl_period(
    climate_data: xr.DataArray, simulation: str, gwl: float
) -> xr.DataArray:
    """
    Extract a specific simulation and crop to its GWL period.

    Parameters
    ----------
    climate_data : xr.DataArray
        Climate data with 'simulation' dimension (from load_wrf_climate_data_all_simulations)
    simulation : str
        Model simulation name (e.g., 'ec-earth3')
    gwl : float
        Global warming level (e.g., 0.8, 2.0)

    Returns
    -------
    xr.DataArray
        Climate data for specified simulation and GWL period
    """
    # Get WRF simulation name
    wrf_sim_name = sim_name_dict[simulation]
    model = wrf_sim_name.split("_")[1]
    ensemble_member = wrf_sim_name.split("_")[2]

    # Get GWL crossing period
    start_year, end_year = get_gwl_crossing_period(model, ensemble_member, gwl)

    # Select specific simulation
    data_sim = climate_data.sel(simulation=wrf_sim_name)

    # Crop to exact GWL period
    data_sim = data_sim.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    return data_sim


def compute_regional_climate_average(
    climate_data: xr.DataArray, shapefile_path: str, region_name: Optional[str] = None
) -> xr.Dataset:
    """
    Compute spatial average of climate data over regions defined by shapefile.

    Parameters
    ----------
    climate_data : xr.DataArray
        Climate data with x, y dimensions
    shapefile_path : str
        Path to shapefile with regional boundaries
    region_name : str, optional
        If provided, compute average for single region only.
        Otherwise, compute for all regions in shapefile.

    Returns
    -------
    xr.Dataset
        Regional averages with 'region' dimension
    """
    # Region name mapping (shapefile names -> standard names)
    region_name_dict = {
        "WECC_MTN": "WECC-MTN",
        "WECC_NW": "WECC-NW",
        "WECC_SW": "WECC-SW",
        "IID": "IID",
        "LDWP": "LDWP",
        "NCNC": "NCNC",
        "PG&E": "PGE",
        "SCE": "SCE",
        "SDG&E": "SDGE",
    }

    # Load shapefile
    gdf = gpd.read_file(shapefile_path)

    # Get WRF CRS
    wrf_crs = get_wrf_crs()

    # Reproject shapefile to WRF CRS
    gdf_reprojected = gdf.to_crs(wrf_crs.to_string())

    # Filter to single region if specified
    if region_name is not None:
        gdf_reprojected = gdf_reprojected[gdf_reprojected["name"] == region_name]

    # Compute regional averages
    regional_data = []

    for index, row in gdf_reprojected.iterrows():
        region_shapefile = row["name"]  # Original shapefile name
        geom = [mapping(row.geometry)]

        # Map to standardized region name
        region_standard = region_name_dict.get(region_shapefile, region_shapefile)

        # Clip to region
        regional_climate = climate_data.rio.clip(
            geom,
            crs=wrf_crs,
            drop=True,
            all_touched=True,
        )

        # Compute spatial mean
        regional_climate = regional_climate.mean(["x", "y"])

        # Add region coordinate with standardized name
        regional_climate = regional_climate.expand_dims({"region": [region_standard]})

        regional_data.append(regional_climate)

    # Combine all regions
    regional_ds = xr.concat(regional_data, dim="region")

    return regional_ds


def extract_point_data(
    climate_data: xr.DataArray,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    x_idx: Optional[int] = None,
    y_idx: Optional[int] = None,
) -> xr.DataArray:
    """
    Extract climate data at a specific point location.

    Parameters
    ----------
    climate_data : xr.DataArray
        Climate data with x, y dimensions
    lat : float, optional
        Target latitude. Required if x_idx/y_idx not provided.
    lon : float, optional
        Target longitude. Required if x_idx/y_idx not provided.
    x_idx : int, optional
        Grid x-index. If provided with y_idx, lat/lon are ignored.
    y_idx : int, optional
        Grid y-index. If provided with x_idx, lat/lon are ignored.

    Returns
    -------
    xr.DataArray
        Climate data at the specified point

    Examples
    --------
    >>> # Using lat/lon
    >>> point_data = extract_point_data(climate_data, lat=34.0, lon=-118.0)
    >>>
    >>> # Using grid indices
    >>> point_data = extract_point_data(climate_data, x_idx=66, y_idx=116)
    """
    if x_idx is not None and y_idx is not None:
        # Use provided indices
        return climate_data.isel(x=x_idx, y=y_idx)

    elif lat is not None and lon is not None:
        # Find nearest valid grid cell
        x_idx, y_idx = find_nearest_valid_point(climate_data, lat, lon)
        return climate_data.isel(x=x_idx, y=y_idx)

    else:
        raise ValueError(
            "Must provide either (lat, lon) or (x_idx, y_idx) for point extraction"
        )


def find_nearest_valid_point(
    data: xr.DataArray, target_lat: float, target_lon: float
) -> Tuple[int, int]:
    """
    Find the nearest non-NaN grid cell to the target coordinates.

    Parameters
    ----------
    data : xr.DataArray
        Data array with lat/lon coordinates
    target_lat : float
        Target latitude
    target_lon : float
        Target longitude

    Returns
    -------
    tuple
        (x_idx, y_idx) of the nearest valid grid cell
    """
    # Select first time point to check for valid data
    if "time" in data.dims:
        sample_data = data.isel(time=0)
    else:
        sample_data = data

    # Calculate distance from target point
    lat_dist = np.abs(data.lat - target_lat)
    lon_dist = np.abs(data.lon - target_lon)

    # Combined distance (simple Euclidean in lat/lon space)
    distance = np.sqrt(lat_dist**2 + lon_dist**2)

    # Mask out NaN locations
    distance_masked = distance.where(~np.isnan(sample_data))

    # Find minimum distance
    min_idx = distance_masked.argmin(...)
    y_idx = int(min_idx["y"].values)
    x_idx = int(min_idx["x"].values)

    actual_lat = float(data.lat.isel(y=y_idx, x=x_idx).values)
    actual_lon = float(data.lon.isel(y=y_idx, x=x_idx).values)

    print(f"Target location: ({target_lat:.2f}°N, {target_lon:.2f}°E)")
    print(f"Nearest valid grid cell: ({actual_lat:.2f}°N, {actual_lon:.2f}°E)")
    print(f"Grid indices: x={x_idx}, y={y_idx}")

    return x_idx, y_idx


def compute_generation_sei_at_location(
    resource: str,
    module: str,
    domain: str,
    simulation: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    x_idx: Optional[int] = None,
    y_idx: Optional[int] = None,
    reference_gwl: float = 0.8,
    window_size: int = 30,
    variable: str = "cf",
    frequency: str = "day",
) -> xr.DataArray:
    """
    Compute SEI for renewable generation at a specific location.

    Parameters
    ----------
    resource : str
        Renewable resource type ('pv', 'windpower')
    module : str
        Resource module ('utility', 'onshore', 'offshore')
    domain : str
        WRF domain ('d02' or 'd03')
    simulation : str
        Model simulation name (e.g., 'ec-earth3')
    lat : float, optional
        Target latitude. Required if x_idx/y_idx not provided.
    lon : float, optional
        Target longitude. Required if x_idx/y_idx not provided.
    x_idx : int, optional
        Grid x-index. If provided with y_idx, lat/lon are ignored.
    y_idx : int, optional
        Grid y-index. If provided with x_idx, lat/lon are ignored.
    reference_gwl : float, optional
        Reference GWL for defining normal conditions, default 0.8
    window_size : int, optional
        Rolling window size for SEI calculation (days), default 30
    variable : str, optional
        Variable to analyze, default 'cf' (capacity factor)
    frequency : str, optional
        Temporal frequency, default 'day'

    Returns
    -------
    xr.DataArray
        SEI with dimensions (dayofyear, year)

    Examples
    --------
    >>> # Using lat/lon
    >>> sei = compute_generation_sei_at_location(
    ...     'pv', 'utility', 'd02', 'ec-earth3',
    ...     lat=34.0, lon=-118.0
    ... )
    >>>
    >>> # Using grid indices
    >>> sei = compute_generation_sei_at_location(
    ...     'windpower', 'onshore', 'd02', 'miroc6',
    ...     x_idx=66, y_idx=116
    ... )
    """
    # Load full timeseries for simulation
    ren_data = get_ren_data_concat(
        resource=resource,
        module=module,
        domain=domain,
        variable=variable,
        frequency=frequency,
        simulation=simulation,
    )

    # Extract point location
    if x_idx is not None and y_idx is not None:
        location_ts = ren_data.isel(x=x_idx, y=y_idx)
    elif lat is not None and lon is not None:
        x_idx, y_idx = find_nearest_valid_point(ren_data, lat, lon)
        location_ts = ren_data.isel(x=x_idx, y=y_idx)
    else:
        raise ValueError(
            "Must provide either (lat, lon) or (x_idx, y_idx) for point extraction"
        )

    # Get reference period for this simulation
    wrf_sim_name = sim_name_dict[simulation]
    model = wrf_sim_name.split("_")[1]
    ensemble_member = wrf_sim_name.split("_")[2]

    ref_start, ref_end = get_gwl_crossing_period(model, ensemble_member, reference_gwl)

    # Extract reference period
    ref_data = location_ts.sel(time=slice(f"{ref_start}", f"{ref_end}"))

    # Compute SEI
    sei = compute_sei(
        ref_data, location_ts, window_size=window_size, fill_missing_year=True
    )

    return sei


def load_generation_sei_from_file(
    filepath: Union[str, Path], convert_to_timeseries: bool = True
) -> xr.DataArray:
    """
    Load pre-computed generation SEI from NetCDF file.

    Parameters
    ----------
    filepath : str or Path
        Path to NetCDF file containing SEI data
    convert_to_timeseries : bool, optional
        If True and data has (dayofyear, year) dimensions, convert to single time dimension.
        Default is True.

    Returns
    -------
    xr.DataArray
        SEI data

    Examples
    --------
    >>> sei = load_generation_sei_from_file("data/location_analysis/sei_ec-earth3.nc")
    """
    sei_data = xr.open_dataarray(filepath)

    # Check if we need to convert from (dayofyear, year) to time dimension
    if (
        convert_to_timeseries
        and "dayofyear" in sei_data.dims
        and "year" in sei_data.dims
    ):
        from sei_computation import stack_doy_year_to_time

        sei_data = stack_doy_year_to_time(sei_data, calendar="noleap")

    return sei_data


def compute_anomalies(
    data: xr.DataArray, reference_data: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """
    Calculate day-of-year anomalies.

    Parameters
    ----------
    data : xr.DataArray
        Data to compute anomalies for
    reference_data : xr.DataArray, optional
        Reference period data for calculating climatology.
        If None, uses data itself to calculate climatology.

    Returns
    -------
    xr.DataArray
        Anomalies relative to day-of-year climatology
    """
    if reference_data is None:
        reference_data = data

    # Calculate day-of-year climatology from reference
    climatology = reference_data.groupby("time.dayofyear").mean("time", skipna=True)

    # Calculate anomalies
    anomalies = data.groupby("time.dayofyear") - climatology

    return anomalies


def merge_for_correlation(
    data1: xr.DataArray, data2: xr.DataArray, var1_name: str, var2_name: str
) -> xr.Dataset:
    """
    Merge two datasets on common dimensions for correlation analysis.

    Parameters
    ----------
    data1 : xr.DataArray
        First variable data
    data2 : xr.DataArray
        Second variable data
    var1_name : str
        Name for first variable in merged dataset
    var2_name : str
        Name for second variable in merged dataset

    Returns
    -------
    xr.Dataset
        Merged dataset with both variables
    """
    # Rename variables
    ds1 = data1.to_dataset(name=var1_name)
    ds2 = data2.to_dataset(name=var2_name)

    # Merge on common dimensions
    merged = xr.merge([ds1, ds2], join="inner")

    return merged


def prepare_correlation_data(
    demand_data: xr.DataArray,
    climate_data: xr.DataArray,
    demand_use_anomaly: bool = True,
    climate_use_anomaly: bool = True,
    demand_var_name: str = "demand",
    climate_var_name: str = "climate",
) -> xr.Dataset:
    """
    Prepare demand and climate data for correlation analysis.

    Handles anomaly calculation and merging with flexible options
    for using raw values or anomalies for each variable.

    Parameters
    ----------
    demand_data : xr.DataArray
        Demand data (SEI or raw peak load)
    climate_data : xr.DataArray
        Climate variable data
    demand_use_anomaly : bool, optional
        If True and demand_data is raw (not SEI), compute anomalies.
        Default True.
    climate_use_anomaly : bool, optional
        If True, compute climate anomalies. Default True.
    demand_var_name : str, optional
        Name for demand variable in output
    climate_var_name : str, optional
        Name for climate variable in output

    Returns
    -------
    xr.Dataset
        Dataset with both variables ready for correlation
    """
    # Process demand data
    if demand_use_anomaly and "sei" not in demand_var_name.lower():
        # Assume raw demand, compute anomalies
        demand_processed = compute_anomalies(demand_data)
    else:
        # Already SEI or user wants raw values
        demand_processed = demand_data

    # Process climate data
    if climate_use_anomaly:
        climate_processed = compute_anomalies(climate_data)
    else:
        climate_processed = climate_data

    # Merge
    merged = merge_for_correlation(
        demand_processed, climate_processed, demand_var_name, climate_var_name
    )

    return merged


def load_demand_sei(
    simulation: str,
    data_dir: str = "../../data/SEI",
    domain: str = "d02",
    reference_gwl: float = 0.8,
) -> xr.DataArray:
    """
    Load demand SEI data from file.

    Parameters
    ----------
    simulation : str
        Model simulation name (e.g., 'ec-earth3')
    data_dir : str, optional
        Directory containing SEI files
    domain : str, optional
        Domain identifier
    reference_gwl : float, optional
        Reference GWL used for SEI calculation

    Returns
    -------
    xr.DataArray
        Demand SEI data
    """
    data_path = Path(data_dir)
    filename = f"demand_reference_{domain}_load_{simulation}_gwlref{reference_gwl}_regional_SEI.nc"
    filepath = data_path / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Demand SEI file not found: {filepath}")

    sei_data = xr.open_dataarray(filepath)

    return sei_data


def load_demand_peak_load(
    simulation: str,
    data_dir: str = "../../data/demand/DailyPeaks",
    gwl: Optional[float] = None,
) -> xr.DataArray:
    """
    Load raw daily peak load data.

    Parameters
    ----------
    simulation : str
        Model simulation name (e.g., 'ec-earth3')
    data_dir : str, optional
        Directory containing peak load CSVs
    gwl : float, optional
        If provided, crop to GWL period

    Returns
    -------
    xr.DataArray
        Daily peak load by region
    """
    import os

    data_path = Path(data_dir)

    # Load all regional CSV files
    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

    region_datasets = []

    for file in csv_files:
        region_name = file.split("_")[0]
        df = pd.read_csv(data_path / file)
        df = df.rename(columns={"Date": "time"})
        df.set_index("time", inplace=True)
        ds = df.to_xarray()

        # Add region coordinate
        ds = ds.expand_dims({"region": [region_name]})
        region_datasets.append(ds)

    # Concatenate all regions
    peak_demand_ds = xr.concat(region_datasets, dim="region")

    # Convert time to datetime
    peak_demand_ds["time"] = pd.to_datetime(peak_demand_ds.time.values)
    peak_demand_ds = peak_demand_ds.convert_calendar("noleap")

    # Rename simulation if needed (EC_EARTH3 -> ec-earth3, etc.)
    sim_rename = {
        "EC_EARTH3": "ec-earth3",
        "MIROC6": "miroc6",
        "MPI_ESM1_2_HR": "mpi-esm1-2-hr",
        "TAIESM1": "taiesm1",
    }

    peak_demand_ds = peak_demand_ds.rename(
        {k: v for k, v in sim_rename.items() if k in peak_demand_ds.data_vars}
    )

    # Select simulation
    if simulation in peak_demand_ds.data_vars:
        demand_data = peak_demand_ds[simulation]
    else:
        raise ValueError(f"Simulation {simulation} not found in demand data")

    # Crop to GWL period if requested
    if gwl is not None:
        wrf_sim_name = sim_name_dict[simulation]
        model = wrf_sim_name.split("_")[1]
        ensemble_member = wrf_sim_name.split("_")[2]
        start_year, end_year = get_gwl_crossing_period(model, ensemble_member, gwl)
        demand_data = demand_data.sel(
            time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
        )

    return demand_data
