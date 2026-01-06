#!/usr/bin/env python3
"""
================================================================================
NASA POWER API Data Fetcher
================================================================================

Fetches solar irradiance and weather data from NASA POWER API.
No API key required - free tier with ~30 requests/minute rate limit.

Provides two modes:
  - Hourly: W/m² irradiance, compatible with physics model Perez calculations
  - Daily: kWh/m²/day totals, for quick energy estimates

NASA POWER Parameters:
  - ALLSKY_SFC_SW_DWN = GHI (Global Horizontal Irradiance)
  - ALLSKY_SFC_SW_DIFF = DHI (Diffuse Horizontal Irradiance)
  - ALLSKY_SFC_SW_DNI = DNI (Direct Normal Irradiance)
  - T2M = 2m Air Temperature (°C)
  - WS2M = 2m Wind Speed (m/s)

USAGE:
------
  # As a module
  from fetch_nasa_power import fetch_nasa_power_hourly, fetch_nasa_power_daily
  
  df = fetch_nasa_power_hourly(lat=8.34, lon=80.38, start="20251201", end="20251215")
  
  # As a standalone script
  python fetch_nasa_power.py --start 20251201 --end 20251215 --mode hourly

================================================================================
"""

import argparse
import requests
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

# =====================================================================
# CONFIGURATION
# =====================================================================

def load_config(config_path=None):
    """Load plant configuration from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "plant_config.json"
    
    with open(config_path, "r") as f:
        return json.load(f)

# =====================================================================
# NASA POWER API ENDPOINTS
# =====================================================================

NASA_POWER_HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Parameter mapping
NASA_PARAMS = {
    "ghi": "ALLSKY_SFC_SW_DWN",
    "dhi": "ALLSKY_SFC_SW_DIFF", 
    "dni": "ALLSKY_SFC_SW_DNI",
    "temp": "T2M",
    "wind": "WS2M"
}

# =====================================================================
# HOURLY DATA FETCHER (W/m²)
# =====================================================================

def fetch_nasa_power_hourly(lat, lon, start_date, end_date, timeout=60):
    """
    Fetch hourly irradiance and weather data from NASA POWER.
    
    Parameters:
    -----------
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees
    start_date : str
        Start date in YYYYMMDD format
    end_date : str
        End date in YYYYMMDD format
    timeout : int
        Request timeout in seconds
    
    Returns:
    --------
    pd.DataFrame with columns: ghi, dni, dhi, air_temp, wind_speed, period_end
        - ghi, dni, dhi in W/m² (instantaneous averages)
        - air_temp in °C
        - wind_speed in m/s
        - period_end as UTC datetime
    
    Raises:
    -------
    requests.HTTPError if API request fails
    ValueError if data is invalid
    """
    print(f"\n{'='*60}")
    print("NASA POWER HOURLY DATA FETCH")
    print(f"{'='*60}")
    print(f"Location: lat={lat}, lon={lon}")
    print(f"Period: {start_date} → {end_date}")
    
    # Build request parameters
    params = {
        "parameters": ",".join([
            NASA_PARAMS["ghi"],
            NASA_PARAMS["dhi"],
            NASA_PARAMS["dni"],
            NASA_PARAMS["temp"],
            NASA_PARAMS["wind"]
        ]),
        "community": "RE",  # Renewable Energy community
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    
    print("\nRequesting hourly data from NASA POWER...")
    resp = requests.get(NASA_POWER_HOURLY_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    
    data = resp.json()
    
    # Check for API errors
    if "messages" in data:
        for msg in data.get("messages", []):
            print(f"  ⚠️ API message: {msg}")
    
    # Extract parameters
    if "properties" not in data or "parameter" not in data["properties"]:
        raise ValueError(f"Invalid API response structure: {list(data.keys())}")
    
    params_data = data["properties"]["parameter"]
    
    # Get timestamps from first parameter
    first_param = list(params_data.values())[0]
    timestamps = list(first_param.keys())
    
    # Build DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "ghi": [params_data[NASA_PARAMS["ghi"]].get(t, np.nan) for t in timestamps],
        "dhi": [params_data[NASA_PARAMS["dhi"]].get(t, np.nan) for t in timestamps],
        "dni": [params_data[NASA_PARAMS["dni"]].get(t, np.nan) for t in timestamps],
        "air_temp": [params_data[NASA_PARAMS["temp"]].get(t, np.nan) for t in timestamps],
        "wind_speed": [params_data[NASA_PARAMS["wind"]].get(t, np.nan) for t in timestamps],
    })
    
    # Parse timestamps (NASA POWER format: YYYYMMDDHH)
    df["period_end"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H")
    df["period_end"] = df["period_end"] + pd.Timedelta(hours=1)  # End of period
    df["period_end"] = df["period_end"].dt.tz_localize("UTC")
    
    # Clean data - NASA POWER uses -999 for missing values
    for col in ["ghi", "dhi", "dni", "air_temp", "wind_speed"]:
        df[col] = df[col].replace(-999, np.nan)
        df[col] = df[col].replace(-999.0, np.nan)
    
    # Clip negative irradiance to 0
    for col in ["ghi", "dhi", "dni"]:
        df[col] = df[col].clip(lower=0)
    
    # Add period column for Solcast compatibility
    df["period"] = "PT60M"
    
    # Reorder columns to match Solcast format
    df = df[["air_temp", "dhi", "dni", "ghi", "wind_speed", "period_end", "period"]]
    
    # Sort by time
    df = df.sort_values("period_end").reset_index(drop=True)
    
    # Validation
    valid_count = df[["ghi", "dhi", "dni"]].notna().all(axis=1).sum()
    total_count = len(df)
    
    print(f"\n✓ Fetched {total_count} hourly records")
    print(f"  Valid irradiance: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
    print(f"  Date range: {df['period_end'].min()} to {df['period_end'].max()}")
    
    # Sample output
    print("\nSample data (first 5 rows):")
    print(df.head().to_string(index=False))
    
    return df


# =====================================================================
# DAILY DATA FETCHER (kWh/m²/day)
# =====================================================================

def fetch_nasa_power_daily(lat, lon, start_date, end_date, timeout=60):
    """
    Fetch daily irradiance and weather data from NASA POWER.
    
    Parameters:
    -----------
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees
    start_date : str
        Start date in YYYYMMDD format
    end_date : str
        End date in YYYYMMDD format
    timeout : int
        Request timeout in seconds
    
    Returns:
    --------
    pd.DataFrame with columns:
        - date: Date index
        - ghi_kwh_m2_day: Daily GHI in kWh/m²/day
        - dhi_kwh_m2_day: Daily DHI in kWh/m²/day
        - dni_kwh_m2_day: Daily DNI in kWh/m²/day
        - temp_c_daily: Daily average temperature in °C
        - wind_mps_daily: Daily average wind speed in m/s
    
    Raises:
    -------
    requests.HTTPError if API request fails
    ValueError if data is invalid
    """
    print(f"\n{'='*60}")
    print("NASA POWER DAILY DATA FETCH")
    print(f"{'='*60}")
    print(f"Location: lat={lat}, lon={lon}")
    print(f"Period: {start_date} → {end_date}")
    
    # Build request parameters
    params = {
        "parameters": ",".join([
            NASA_PARAMS["ghi"],
            NASA_PARAMS["dhi"],
            NASA_PARAMS["dni"],
            NASA_PARAMS["temp"],
            NASA_PARAMS["wind"]
        ]),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    
    print("\nRequesting daily data from NASA POWER...")
    resp = requests.get(NASA_POWER_DAILY_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    
    data = resp.json()
    
    # Check for API errors
    if "messages" in data:
        for msg in data.get("messages", []):
            print(f"  ⚠️ API message: {msg}")
    
    # Extract parameters
    if "properties" not in data or "parameter" not in data["properties"]:
        raise ValueError(f"Invalid API response structure: {list(data.keys())}")
    
    params_data = data["properties"]["parameter"]
    
    # Get dates from first parameter
    first_param = list(params_data.values())[0]
    dates = list(first_param.keys())
    
    # Build DataFrame
    df = pd.DataFrame({
        "date": pd.to_datetime(dates, format="%Y%m%d"),
        "ghi_kwh_m2_day": [params_data[NASA_PARAMS["ghi"]].get(d, np.nan) for d in dates],
        "dhi_kwh_m2_day": [params_data[NASA_PARAMS["dhi"]].get(d, np.nan) for d in dates],
        "dni_kwh_m2_day": [params_data[NASA_PARAMS["dni"]].get(d, np.nan) for d in dates],
        "temp_c_daily": [params_data[NASA_PARAMS["temp"]].get(d, np.nan) for d in dates],
        "wind_mps_daily": [params_data[NASA_PARAMS["wind"]].get(d, np.nan) for d in dates],
    })
    
    # Clean data - NASA POWER uses -999 for missing values
    for col in df.columns:
        if col != "date":
            df[col] = df[col].replace(-999, np.nan)
            df[col] = df[col].replace(-999.0, np.nan)
    
    # Clip negative irradiance to 0
    for col in ["ghi_kwh_m2_day", "dhi_kwh_m2_day", "dni_kwh_m2_day"]:
        df[col] = df[col].clip(lower=0)
    
    # Set index and sort
    df = df.set_index("date").sort_index()
    
    # Validation
    valid_count = df[["ghi_kwh_m2_day", "dhi_kwh_m2_day", "dni_kwh_m2_day"]].notna().all(axis=1).sum()
    total_count = len(df)
    
    print(f"\n✓ Fetched {total_count} daily records")
    print(f"  Valid irradiance: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # Sample output
    print("\nSample data (first 5 rows):")
    print(df.head().to_string())
    
    # Sanity check
    assert (df[["ghi_kwh_m2_day", "dhi_kwh_m2_day", "dni_kwh_m2_day"]].dropna() >= 0).all().all(), \
        "Negative irradiance values detected!"
    
    return df


# =====================================================================
# SAVE TO CSV (Solcast-compatible format)
# =====================================================================

def save_hourly_csv(df, output_path):
    """Save hourly data to CSV in Solcast-compatible format."""
    # Rename columns to match Solcast format exactly
    df_out = df.rename(columns={
        "wind_speed": "wind_speed_10m"  # NASA uses 2m, but we keep column name
    })
    
    # Reorder for Solcast compatibility
    cols = ["air_temp", "dhi", "dni", "ghi", "wind_speed_10m", "period_end", "period"]
    df_out = df_out[cols]
    
    df_out.to_csv(output_path, index=False)
    print(f"\n✓ Saved hourly data to: {output_path}")
    

def save_daily_csv(df, output_path):
    """Save daily data to CSV."""
    df.to_csv(output_path)
    print(f"\n✓ Saved daily data to: {output_path}")


# =====================================================================
# CLI INTERFACE
# =====================================================================

def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Fetch solar irradiance data from NASA POWER API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_nasa_power.py --start 20251201 --end 20251215 --mode hourly
  python fetch_nasa_power.py --start 20251201 --end 20251215 --mode daily
  python fetch_nasa_power.py --start 20251201 --end 20251215 --mode both
        """
    )
    
    parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYYMMDD format"
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYYMMDD format"
    )
    parser.add_argument(
        "--mode",
        choices=["hourly", "daily", "both"],
        default="hourly",
        help="Data mode: hourly (W/m²), daily (kWh/m²/day), or both"
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Latitude (default: from plant_config.json)"
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Longitude (default: from plant_config.json)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ../../data/)"
    )
    
    args = parser.parse_args()
    
    # Load config for defaults
    config = load_config()
    lat = args.lat or config["location"]["lat"]
    lon = args.lon or config["location"]["lon"]
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(__file__).parent.parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data
    if args.mode in ["hourly", "both"]:
        df_hourly = fetch_nasa_power_hourly(lat, lon, args.start, args.end)
        save_hourly_csv(df_hourly, output_dir / "nasa_power_hourly.csv")
    
    if args.mode in ["daily", "both"]:
        df_daily = fetch_nasa_power_daily(lat, lon, args.start, args.end)
        save_daily_csv(df_daily, output_dir / "nasa_power_daily.csv")
    
    print(f"\n{'='*60}")
    print("NASA POWER DATA FETCH COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


