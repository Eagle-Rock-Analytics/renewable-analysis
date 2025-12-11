#!/usr/bin/env python
"""
Create drought mask files from renewable energy data and threshold files.

This script loads renewable energy capacity factor data, compares it against
pre-computed drought thresholds, and creates a binary drought mask file.

Usage:
    python create_drought_mask.py --resource pv --module utility --simulation taiesm1 --reference_gwl 0.8
    python create_drought_mask.py -r windpower -m onshore -s ec-earth3 -g 1.5
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir / "src"))

from src.renewable_data_load import get_ren_data_concat


def create_drought_mask(
    resource,
    module,
    simulation,
    reference_gwl,
    domain="d02",
    variable="cf",
    frequency="day",
    data_dir=None,
):
    """
    Create a binary drought mask from renewable energy data and thresholds.

    Parameters
    ----------
    resource : str
        Resource type ('pv' or 'windpower')
    module : str
        Module type ('utility', 'distributed', 'onshore', 'offshore')
    simulation : str
        Climate simulation name (e.g., 'taiesm1', 'ec-earth3', 'miroc6', 'mpi-esm1-2-hr')
    reference_gwl : float
        Reference Global Warming Level (e.g., 0.8, 1.5, 2.0)
    domain : str, optional
        Spatial domain ('d02' or 'd03'), default 'd02'
    variable : str, optional
        Variable name ('cf' for capacity factor or 'gen' for generation), default 'cf'
    frequency : str, optional
        Temporal frequency ('day' or '1hr'), default 'day'
    data_dir : str or Path, optional
        Base directory for data files. If None, uses '../data' relative to script location.

    Returns
    -------
    str
        Path to the created drought mask file
    """
    print(f"\n{'='*70}")
    print(f"Creating drought mask for:")
    print(f"  Resource: {resource}")
    print(f"  Module: {module}")
    print(f"  Simulation: {simulation}")
    print(f"  Reference GWL: {reference_gwl}")
    print(f"  Domain: {domain}")
    print(f"{'='*70}\n")

    # Set up data directory
    if data_dir is None:
        data_dir = project_dir / "data"
    else:
        data_dir = Path(data_dir)

    # Load threshold file
    threshold_file = (
        data_dir
        / "thresholds"
        / f"{resource}_{module}_{domain}_{variable}_{simulation}_gwlref{reference_gwl}_10th_pctile.nc"
    )

    if not threshold_file.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_file}")

    print(f"Loading threshold file: {threshold_file.name}")
    drought_threshold_ds = xr.open_dataset(threshold_file)

    # Load renewable energy data
    print(f"Loading renewable energy data...")
    ren_ds = get_ren_data_concat(
        resource, module, domain, variable, frequency, simulation
    )
    ren_ds = ren_ds.convert_calendar("noleap")
    print(f"  Data shape: {ren_ds.dims}")
    print(f"  Time range: {ren_ds.time.values[0]} to {ren_ds.time.values[-1]}")

    # Reshape array by day of year
    print(f"Reshaping data by day of year...")
    ds_doy = ren_ds.copy(deep=True)
    ds_doy["dayofyear"] = ds_doy.time.dt.dayofyear
    ds_doy["year"] = ds_doy.time.dt.year
    ds_doy = ds_doy.assign_coords(
        {"dayofyear": ds_doy.time.dt.dayofyear, "year": ds_doy.time.dt.year}
    )
    # Reshape time dimension
    ds_doy = ds_doy.drop_vars("time").set_index(time=["dayofyear", "year"]).unstack()

    # Calculate drought metric (difference from threshold)
    print(f"Calculating drought metric (values below threshold)...")
    drought_ds = (ds_doy - drought_threshold_ds.reference_gen).load()

    # Reshape back into daily timeseries
    print(f"Reshaping back to time series...")
    drought_ds = drought_ds.stack(time=["year", "dayofyear"])
    drought_ds = drought_ds.reset_index("time").assign_coords(time=ren_ds.time)
    drought_ds = drought_ds.load()

    # Create binary drought mask
    # 1 where drought_ds < 0 (below threshold), 0 otherwise
    print(f"Creating binary drought mask...")
    drought_mask = xr.where(drought_ds < 0, 1, 0)
    drought_mask.name = "drought_mask"
    drought_mask = drought_mask.load()

    # Calculate statistics
    total_points = drought_mask.size
    drought_points = (drought_mask == 1).sum().values
    drought_fraction = drought_points / total_points * 100
    print(f"\nDrought Statistics:")
    print(f"  Total data points: {total_points:,}")
    print(f"  Drought points: {drought_points:,}")
    print(f"  Drought fraction: {drought_fraction:.2f}%")

    # Save drought mask to file
    output_dir = data_dir / "drought_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_output_file = (
        output_dir
        / f"{resource}_{module}_{domain}_{variable}_{simulation}_gwlref{reference_gwl}_drought_mask_only.nc"
    )

    # Add metadata to drought mask
    drought_mask.attrs["resource"] = resource
    drought_mask.attrs["module"] = module
    drought_mask.attrs["domain"] = domain
    drought_mask.attrs["variable"] = variable
    drought_mask.attrs["simulation"] = simulation
    drought_mask.attrs["reference_gwl"] = float(reference_gwl)
    drought_mask.attrs["description"] = (
        "Binary drought mask: 1 = drought (below threshold), 0 = no drought"
    )
    drought_mask.attrs["threshold_file"] = str(threshold_file.name)

    # Save with appropriate encoding
    print(f"\nSaving drought mask to: {mask_output_file.name}")
    encoding = {"drought_mask": {"dtype": "int32", "_FillValue": -999}}
    drought_mask.to_netcdf(mask_output_file, encoding=encoding, format="NETCDF4")

    file_size_mb = mask_output_file.stat().st_size / 1e6
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"\n✓ Successfully created drought mask file!")
    print(f"{'='*70}\n")

    return str(mask_output_file)


def main():
    """Parse command line arguments and create drought mask."""
    parser = argparse.ArgumentParser(
        description="Create binary drought mask from renewable energy data and thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create drought mask for PV utility-scale, TaiESM1 simulation, GWL 0.8
  python create_drought_mask.py --resource pv --module utility --simulation taiesm1 --reference_gwl 0.8

  # Create drought mask for onshore wind, EC-Earth3 simulation, GWL 1.5
  python create_drought_mask.py -r windpower -m onshore -s ec-earth3 -g 1.5

  # Create drought mask for distributed PV, MIROC6 simulation, GWL 2.0
  python create_drought_mask.py -r pv -m distributed -s miroc6 -g 2.0 -d d03
        """,
    )

    # Required arguments
    parser.add_argument(
        "-r",
        "--resource",
        required=True,
        choices=["pv", "windpower"],
        help="Resource type",
    )
    parser.add_argument(
        "-m",
        "--module",
        required=True,
        choices=["utility", "distributed", "onshore", "offshore"],
        help="Module type",
    )
    parser.add_argument(
        "-s",
        "--simulation",
        required=True,
        choices=["taiesm1", "ec-earth3", "miroc6", "mpi-esm1-2-hr", "era5"],
        help="Climate simulation name",
    )
    parser.add_argument(
        "-g",
        "--reference_gwl",
        required=True,
        type=float,
        help="Reference Global Warming Level (e.g., 0.8, 1.5, 2.0)",
    )

    # Optional arguments
    parser.add_argument(
        "-d",
        "--domain",
        default="d02",
        choices=["d02", "d03"],
        help="Spatial domain (default: d02)",
    )
    parser.add_argument(
        "-v",
        "--variable",
        default="cf",
        choices=["cf", "gen"],
        help="Variable name (default: cf)",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        default="day",
        choices=["day", "1hr"],
        help="Temporal frequency (default: day)",
    )
    parser.add_argument(
        "--data_dir", type=str, help="Base directory for data files (default: ../data)"
    )

    args = parser.parse_args()

    try:
        output_file = create_drought_mask(
            resource=args.resource,
            module=args.module,
            simulation=args.simulation,
            reference_gwl=args.reference_gwl,
            domain=args.domain,
            variable=args.variable,
            frequency=args.frequency,
            data_dir=args.data_dir,
        )
        print(f"Output file: {output_file}")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
