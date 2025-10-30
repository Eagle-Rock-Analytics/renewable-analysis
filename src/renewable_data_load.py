import numpy as np
import pandas as pd
import xarray as xr
from climakitae.core.paths import GWL_1850_1900_FILE
from climakitae.util.utils import read_csv_file

ae_wrf_sims = [
    "WRF_CESM2_r11i1p1f1",
    "WRF_MIROC6_r1i1p1f1",
    "WRF_CNRM-ESM2-1_r1i1p1f2",
    "WRF_EC-Earth3_r1i1p1f1",
    "WRF_TaiESM1_r1i1p1f1",
    "WRF_MPI-ESM1-2-HR_r3i1p1f1",
    "WRF_EC-Earth3-Veg_r1i1p1f1",
    "WRF_FGOALS-g3_r1i1p1f1",
]


ren_sims = ["ec-earth3", "mpi-esm1-2-hr", "miroc6", "taiesm1"]

sim_name_dict = {
    "ec-earth3": "WRF_EC-Earth3_r1i1p1f1",
    "mpi-esm1-2-hr": "WRF_MPI-ESM1-2-HR_r3i1p1f1",
    "miroc6": "WRF_MIROC6_r1i1p1f1",
    "taiesm1": "WRF_TaiESM1_r1i1p1f1",
}

sim_gwl_name_dict = {
    "ec-earth3": "WRF_EC-Earth3_r1i1p1f1_historical+ssp370",
    "mpi-esm1-2-hr": "WRF_MPI-ESM1-2-HR_r3i1p1f1_historical+ssp370",
    "miroc6": "WRF_MIROC6_r1i1p1f1_historical+ssp370",
    "taiesm1": "WRF_TaiESM1_r1i1p1f1_historical+ssp370",
}

scenario_dict = {
    "historical": "Historical Climate",
    "ssp370": "SSP 3-7.0",
    "reanalysis": "Historical Reconstruction",
}
resolution_dict = {"d03": "3 km", "d02": "9 km"}

frequency_dict = {"day", "daily", "1hr", "hourly"}


gwl_name_dict = {
    "plus08c": 0.8,
    "plus10c": 1.0,
    "plus12c": 1.2,
    "plus15c": 1.5,
    "plus20c": 2.0,
    "plus25c": 2.5,
    "plus30c": 3.0,
    "plus40c": 4.0,
}


def get_ren_cf_data(
    resource, module, domain, variable, frequency, simulation, scenario
):
    path = f"s3://wfclimres/era/{resource}_{module}/{simulation}/{scenario}/{frequency}/{variable}/{domain}/"
    ds = xr.open_zarr(path, storage_options={"anon": True})
    ds = ds[variable]
    ds = ds.isel(
        x=slice(10, -10), y=slice(10, -10)
    )  # trim the edges to match the WRF AE domain
    return ds


def get_ren_drought_data(
    resource, module, domain, variable, frequency, simulation, gwl
):
    path = f"s3://wfclimres/era/resource_drought/{resource}/{resource}_{module}/{simulation}/{gwl}/{frequency}/{variable}/{domain}"
    ds = xr.open_zarr(path, storage_options={"anon": True})
    ds = ds[variable]
    ds = ds.isel(
        x=slice(10, -10), y=slice(10, -10)
    )  # trim the edges to match the WRF AE domain
    return ds


def get_ren_cf_data_gwl(resource, module, domain, variable, frequency, simulation, gwl):
    gwl_arr = np.array(gwl)

    # historical
    path = f"s3://wfclimres/era/{resource}_{module}/{simulation}/historical/{frequency}/{variable}/{domain}/"
    hist_ds = xr.open_zarr(path, storage_options={"anon": True})

    path = f"s3://wfclimres/era/{resource}_{module}/{simulation}/ssp370/{frequency}/{variable}/{domain}/"
    fut_ds = xr.open_zarr(path, storage_options={"anon": True})

    # combine historical and future
    ds = xr.concat([hist_ds, fut_ds], dim="time")

    ds = ds["cf"]
    ds = ds.isel(
        x=slice(10, -10), y=slice(10, -10)
    )  # trim the edges to match the WRF AE domain

    WRF_sim_name = sim_name_dict[simulation]
    model = WRF_sim_name.split("_")[1]
    ensemble_member = WRF_sim_name.split("_")[2]

    gwl_list = []
    for gwl in gwl_arr:
        # get bounds for gwl
        start_year, end_year = get_gwl_crossing_period(model, ensemble_member, gwl)
        print(f"{simulation} {gwl}C: {start_year}-{end_year}")
        # select data for gwl period
        # gwl_subset = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        gwl_subset = ds.sel(time=slice(f"{start_year}", f"{end_year}"))

        gwl_subset = gwl_subset.expand_dims({"warming_level": [gwl]})
        gwl_list.append(gwl_subset)

    # This does not work well as a merge or a concat because of the time and gwl dimensions both being there. looking into this.
    # comb_ds = xr.merge(gwl_list)
    return gwl_list


def get_gwl_crossing_period(
    model: str, ensemble_member: str, warming_level: float
) -> tuple[int, int] | None:
    """
    Get the 30-year period where a specific climate simulation crosses a given global warming level.
    Uses pre-calculated GWL crossing times from the 1850-1900 reference period.

    Parameters
    ----------
    model : str
        The climate model name (e.g., 'CESM2')
    ensemble_member : str
        The ensemble member identifier (e.g., 'r11i1p1f1')
    warming_level : float
        The global warming level to find the crossing point for (e.g., 2.0)

    Returns
    -------
    Optional[Tuple[int, int]]
        A tuple of (start_year, end_year) representing the 30-year period centered
        on when the warming level is crossed. Returns None if the warming level is
        never reached or if the model/ensemble combination is not found.

    Examples
    --------
    >>> start_year, end_year = get_gwl_crossing_period('CESM2', 'r11i1p1f1', 2.0)
    >>> print(f"2.0°C warming period: {start_year}-{end_year}")
    """
    # Read the pre-generated GWL crossing table
    df = read_csv_file(GWL_1850_1900_FILE)

    # Filter for the specific model and ensemble member
    mask = (df["GCM"] == model) & (df["run"] == ensemble_member)
    row = df[mask]

    if row.empty:
        return None

    # Get the crossing time for the specified warming level
    # Convert warming level to string to match column names
    wl_str = str(warming_level)
    if wl_str not in row.columns:
        return None

    crossing_time = row[wl_str].iloc[0]
    crossing_time = pd.to_datetime(crossing_time)

    if pd.isna(crossing_time):
        return None

    # Calculate the 30 year period centered on the crossing time
    crossing_year = crossing_time.year
    start_year = crossing_year - 15
    end_year = crossing_year + 14  # +14 to get a total of 30 years

    return (start_year, end_year)
