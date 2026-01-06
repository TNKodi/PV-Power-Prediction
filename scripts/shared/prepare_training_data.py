#!/usr/bin/env python3
"""
================================================================================
PREPARE TRAINING DATA PIPELINE
================================================================================

This script orchestrates the complete data preparation for model training:

1. Fetches NASA POWER hourly data for the specified year
2. Runs the Python physics model to generate "ground truth" predictions
3. Creates a random 80/20 train/test split by DAYS (not hours)
4. Saves split datasets and metadata for reproducibility

No data leakage: Train and test sets are completely separate.

USAGE:
------
  # Prepare data for full year 2025 with 20% test set
  python prepare_training_data.py --year 2025 --test-ratio 0.2

  # Use a specific random seed for reproducibility
  python prepare_training_data.py --year 2025 --test-ratio 0.2 --seed 42

================================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "python_physics"))

from fetch_nasa_power import fetch_nasa_power_hourly

# =====================================================================
# PATHS
# =====================================================================

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
CONFIG_DIR = PROJECT_DIR / "config"

# =====================================================================
# CONFIGURATION
# =====================================================================

def load_config():
    """Load plant configuration."""
    with open(CONFIG_DIR / "plant_config.json") as f:
        return json.load(f)


# =====================================================================
# PHYSICS MODEL (Inline for speed - avoids subprocess overhead)
# =====================================================================

import pvlib
from pvlib.location import Location


def run_physics_model(df_weather, config):
    """
    Run physics-based PV model on weather data.
    
    This is a simplified inline version for batch processing.
    Uses the same physics as physics_model.py but without CLI overhead.
    
    Parameters:
    -----------
    df_weather : pd.DataFrame
        Hourly weather data with columns: ghi, dni, dhi, air_temp, wind_speed
        Index must be UTC DatetimeIndex
    config : dict
        Plant configuration from plant_config.json
    
    Returns:
    --------
    pd.Series with AC power output in kW, indexed by timestamp
    """
    # Extract config values
    lat = config["location"]["lat"]
    lon = config["location"]["lon"]
    altitude = config["location"]["altitude_m"]
    orientations = config["orientations"]
    module_area = config["module"]["area_m2"]
    module_efficiency = config["module"]["efficiency_stc"]
    gamma_p = config["module"]["gamma_p"]
    inv_ac_rating = config["inverter"]["ac_rating_kw"]
    albedo = config["losses"]["albedo"]
    
    # Calculate loss factors
    losses = config["losses"]
    dc_loss_factor = (
        (1 - losses["soiling"]) *
        (1 - losses["lid"]) *
        (1 + losses["module_quality"]) *
        (1 - losses["mismatch"]) *
        (1 - losses["dc_wiring"]) *
        losses["far_shading"]
    )
    ac_wiring_loss = losses["ac_wiring"]
    
    # SAPM thermal model
    sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
    
    # IAM table
    iam_angles = np.array(config["iam"]["angles"])
    iam_values = np.array(config["iam"]["values"])
    
    # Total module area
    total_module_area = sum(o["module_count"] * module_area for o in orientations)
    
    # Site location
    site = Location(lat, lon, altitude=altitude)
    solpos = site.get_solarposition(df_weather.index)
    dni_extra = pvlib.irradiance.get_extra_radiation(df_weather.index)
    
    # Initialize output
    plant_ac = pd.Series(0.0, index=df_weather.index)
    
    for o in orientations:
        tilt = o["tilt"]
        azimuth = o["azimuth"]
        area_fraction = (o["module_count"] * module_area) / total_module_area
        
        # POA irradiance using Perez model
        irr = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solpos["zenith"],
            solar_azimuth=solpos["azimuth"],
            dni=df_weather["dni"],
            ghi=df_weather["ghi"],
            dhi=df_weather["dhi"],
            dni_extra=dni_extra,
            model="perez",
            albedo=albedo
        )
        
        poa = irr["poa_global"].clip(lower=0)
        
        # AOI and IAM
        aoi = pvlib.irradiance.aoi(tilt, azimuth, solpos["zenith"], solpos["azimuth"])
        aoi_clipped = aoi.clip(0, 90)
        iam = np.interp(aoi_clipped, iam_angles, iam_values)
        
        # Effective POA
        poa_optical = poa * iam
        
        # Cell temperature (SAPM)
        cell_temp = pvlib.temperature.sapm_cell(
            poa_global=poa,
            temp_air=df_weather["air_temp"],
            wind_speed=df_weather["wind_speed"],
            a=sapm_params["a"],
            b=sapm_params["b"],
            deltaT=sapm_params["deltaT"]
        )
        
        # DC power per m²
        pdc_kwm2 = poa_optical * module_efficiency / 1000
        
        # Temperature coefficient
        pdc_kwm2_temp = pdc_kwm2 * (1 + gamma_p * (cell_temp - 25))
        
        # Apply DC losses
        pdc_kwm2_eff = pdc_kwm2_temp * dc_loss_factor
        
        # Scale to orientation area
        area_i = total_module_area * area_fraction
        pdc_total = pdc_kwm2_eff * area_i
        
        # Inverter efficiency (simplified flat 98%)
        inv_eff = config["inverter"]["flat_efficiency"]
        pac = pdc_total * inv_eff
        
        plant_ac += pac
    
    # Inverter clipping
    plant_ac = plant_ac.clip(upper=inv_ac_rating)
    
    # AC wiring losses
    plant_ac = plant_ac * (1 - ac_wiring_loss)
    
    return plant_ac


# =====================================================================
# DATA SPLITTING
# =====================================================================

def create_train_test_split(df, test_ratio=0.2, seed=None):
    """
    Split data into train and test sets by DAYS (not individual hours).
    
    This prevents data leakage from the same day appearing in both sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset with DatetimeIndex
    test_ratio : float
        Fraction of days to use for testing (0-1)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    tuple: (train_df, test_df, split_info)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get unique dates
    df_local = df.copy()
    if df_local.index.tz is not None:
        dates = df_local.index.tz_convert("Asia/Colombo").date
    else:
        dates = df_local.index.date
    
    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)
    n_test = int(n_dates * test_ratio)
    
    # Random shuffle and split
    shuffled_indices = np.random.permutation(n_dates)
    test_indices = shuffled_indices[:n_test]
    train_indices = shuffled_indices[n_test:]
    
    test_dates = set(unique_dates[test_indices])
    train_dates = set(unique_dates[train_indices])
    
    # Create masks
    train_mask = pd.Series(dates, index=df.index).isin(train_dates)
    test_mask = pd.Series(dates, index=df.index).isin(test_dates)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    # Split info for reproducibility
    split_info = {
        "seed": seed,
        "test_ratio": test_ratio,
        "total_days": n_dates,
        "train_days": len(train_dates),
        "test_days": len(test_dates),
        "train_hours": len(train_df),
        "test_hours": len(test_df),
        "train_dates": sorted([str(d) for d in train_dates]),
        "test_dates": sorted([str(d) for d in test_dates]),
        "created_at": datetime.now().isoformat()
    }
    
    return train_df, test_df, split_info


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for PV model comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_training_data.py --year 2025 --test-ratio 0.2
  python prepare_training_data.py --year 2025 --test-ratio 0.2 --seed 42
  python prepare_training_data.py --start 20250101 --end 20251231 --test-ratio 0.2
        """
    )
    
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to fetch data for (e.g., 2025)"
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date in YYYYMMDD format (alternative to --year)"
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date in YYYYMMDD format (alternative to --year)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of days to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip NASA POWER fetch if data already exists"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.year:
        start_date = f"{args.year}0101"
        end_date = f"{args.year}1231"
    elif args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        parser.error("Either --year or both --start and --end are required")
    
    print("=" * 70)
    print("PREPARE TRAINING DATA PIPELINE")
    print("=" * 70)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    
    # Load config
    config = load_config()
    lat = config["location"]["lat"]
    lon = config["location"]["lon"]
    
    # =====================================================================
    # STEP 1: FETCH NASA POWER DATA
    # =====================================================================
    
    nasa_power_path = DATA_DIR / f"nasa_power_{start_date[:4]}.csv"
    
    if args.skip_fetch and nasa_power_path.exists():
        print(f"\n[1/4] Loading existing NASA POWER data from {nasa_power_path}...")
        df_weather = pd.read_csv(nasa_power_path)
        df_weather["period_end"] = pd.to_datetime(df_weather["period_end"], utc=True)
        df_weather = df_weather.set_index("period_end")
    else:
        print(f"\n[1/4] Fetching NASA POWER data...")
        
        # NASA POWER has a limit on request size, so fetch in chunks
        # Max ~1 year per request for hourly data
        df_weather = fetch_nasa_power_hourly(lat, lon, start_date, end_date, timeout=120)
        
        # Ensure proper index
        if "period_end" in df_weather.columns:
            df_weather["period_end"] = pd.to_datetime(df_weather["period_end"], utc=True)
            df_weather = df_weather.set_index("period_end")
        
        # Rename columns to match expected format
        if "wind_speed_10m" in df_weather.columns:
            df_weather = df_weather.rename(columns={"wind_speed_10m": "wind_speed"})
        
        # Save raw data
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_weather.to_csv(nasa_power_path)
        print(f"  Saved to: {nasa_power_path}")
    
    print(f"  Total hours: {len(df_weather)}")
    print(f"  Date range: {df_weather.index.min()} to {df_weather.index.max()}")
    
    # Filter out rows with missing data
    required_cols = ["ghi", "dni", "dhi", "air_temp", "wind_speed"]
    missing_before = len(df_weather)
    df_weather = df_weather.dropna(subset=required_cols)
    missing_after = len(df_weather)
    if missing_before > missing_after:
        print(f"  Removed {missing_before - missing_after} rows with missing data")
    
    if len(df_weather) == 0:
        print("\n⚠️  ERROR: No valid data after removing NaN rows!")
        print("   NASA POWER may not have data for the requested date range.")
        print("   Try dates that are at least 7 days in the past.")
        sys.exit(1)
    
    # =====================================================================
    # STEP 2: RUN PHYSICS MODEL
    # =====================================================================
    
    print(f"\n[2/4] Running Python physics model on all data...")
    
    start_time = time.perf_counter()
    ac_power = run_physics_model(df_weather, config)
    physics_time = time.perf_counter() - start_time
    
    print(f"  Processing time: {physics_time:.2f}s ({len(df_weather)/physics_time:.0f} samples/s)")
    print(f"  Total energy: {ac_power.sum():.1f} kWh")
    
    # Combine weather and predictions
    df_combined = df_weather.copy()
    df_combined["ac_power_kw"] = ac_power
    
    # Remove any remaining NaN values (e.g., from physics model edge cases)
    before_count = len(df_combined)
    df_combined = df_combined.dropna(subset=["ac_power_kw"])
    after_count = len(df_combined)
    if before_count > after_count:
        print(f"  Removed {before_count - after_count} rows with NaN predictions")
    
    # =====================================================================
    # STEP 3: CREATE TRAIN/TEST SPLIT
    # =====================================================================
    
    print(f"\n[3/4] Creating train/test split...")
    
    train_df, test_df, split_info = create_train_test_split(
        df_combined,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"  Train set: {split_info['train_days']} days, {split_info['train_hours']} hours")
    print(f"  Test set: {split_info['test_days']} days, {split_info['test_hours']} hours")
    
    # =====================================================================
    # STEP 4: SAVE OUTPUTS
    # =====================================================================
    
    print(f"\n[4/4] Saving outputs...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save combined data with predictions
    combined_path = DATA_DIR / "physics_predictions_all.csv"
    df_combined.to_csv(combined_path)
    print(f"  Saved: {combined_path}")
    
    # Save train data
    train_path = DATA_DIR / "train_data.csv"
    train_df.to_csv(train_path)
    print(f"  Saved: {train_path}")
    
    # Save test data
    test_path = DATA_DIR / "test_data.csv"
    test_df.to_csv(test_path)
    print(f"  Saved: {test_path}")
    
    # Save split info
    split_info_path = OUTPUT_DIR / "split_info.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Saved: {split_info_path}")
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nData files created:")
    print(f"  - {nasa_power_path}")
    print(f"  - {combined_path}")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {split_info_path}")
    print(f"\nNext steps:")
    print(f"  1. Train surrogate model: python fit_surrogate.py --train-data {train_path}")
    print(f"  2. Evaluate models: python evaluate_models.py --test-data {test_path}")
    print("=" * 70)
    
    return train_df, test_df, split_info


if __name__ == "__main__":
    main()

