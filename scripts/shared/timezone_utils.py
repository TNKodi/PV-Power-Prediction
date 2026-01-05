"""
Timezone Utilities for Power Prediction

Provides consistent timezone handling across all scripts:
- Raw Meteocontrol data: Asia/Colombo (GMT +5:30) - implied, not explicit
- Solcast/ERA5 data: UTC
- Internal processing: UTC
- Final outputs: Asia/Colombo (GMT +5:30)

Usage:
------
    from timezone_utils import load_meteocontrol, to_utc, to_local, LOCAL_TZ, UTC

    # Load Meteocontrol data with proper timestamps
    df_actual = load_meteocontrol("../data/meteocontrol_actual.csv")

    # Convert model output to local timezone for saving
    df_output = to_local(df_predictions)
"""

import pandas as pd
from pathlib import Path

# =====================================================================
# TIMEZONE CONSTANTS
# =====================================================================

LOCAL_TZ = "Asia/Colombo"  # GMT +5:30
UTC = "UTC"

# Default year/month for Meteocontrol data (adjust as needed)
# This should match the period of your actual data
DEFAULT_YEAR = 2025
DEFAULT_MONTH = 12


# =====================================================================
# METEOCONTROL LOADER
# =====================================================================

def load_meteocontrol(filepath, year=None, month=None):
    """
    Load Meteocontrol actual generation data with explicit timestamps.
    
    The raw Meteocontrol export contains only day numbers (1-31) in the
    "Category" column. This function adds proper ISO 8601 timestamps
    with the Asia/Colombo timezone.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to meteocontrol_actual.csv
    year : int, optional
        Year for the data (default: 2025)
    month : int, optional
        Month for the data (default: 12)
    
    Returns:
    --------
    pd.DataFrame with columns:
        - timestamp: DatetimeIndex with Asia/Colombo timezone
        - day: int (1-31)
        - actual_kwh: float (daily energy in kWh)
        - Other columns from raw CSV
    
    Example:
    --------
        df = load_meteocontrol("../data/meteocontrol_actual.csv")
        print(df.index)  # DatetimeIndex with Asia/Colombo timezone
    """
    if year is None:
        year = DEFAULT_YEAR
    if month is None:
        month = DEFAULT_MONTH
    
    # Load raw CSV (semicolon-separated)
    df = pd.read_csv(filepath, sep=";")
    
    # Parse day from Category column (handles both string and numeric)
    df["day"] = pd.to_numeric(df["Category"], errors="coerce").astype(int)
    
    # Create explicit timestamps at midnight local time
    # Each row represents a full day's energy, so we use midnight as the timestamp
    timestamps = pd.to_datetime([
        f"{year}-{month:02d}-{day:02d}T00:00:00"
        for day in df["day"]
    ])
    
    # Localize to Asia/Colombo (this is the native timezone of the data)
    df["timestamp"] = timestamps.tz_localize(LOCAL_TZ)
    df = df.set_index("timestamp")
    
    # Rename Power column to actual_kwh for clarity
    if "Power" in df.columns:
        df["actual_kwh"] = pd.to_numeric(df["Power"], errors="coerce")
    
    # Sort by timestamp
    df = df.sort_index()
    
    return df


def load_meteocontrol_utc(filepath, year=None, month=None):
    """
    Load Meteocontrol data and convert to UTC.
    
    Convenience function that loads and immediately converts to UTC
    for internal processing.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to meteocontrol_actual.csv
    year, month : int, optional
        Year and month for the data
    
    Returns:
    --------
    pd.DataFrame with UTC DatetimeIndex
    """
    df = load_meteocontrol(filepath, year, month)
    return to_utc(df)


# =====================================================================
# TIMEZONE CONVERSION
# =====================================================================

def to_utc(df):
    """
    Convert DataFrame index to UTC.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with timezone-aware DatetimeIndex
    
    Returns:
    --------
    pd.DataFrame with UTC DatetimeIndex
    
    Raises:
    -------
    ValueError if index is not timezone-aware
    """
    if df.index.tz is None:
        raise ValueError(
            "DataFrame index is not timezone-aware. "
            "Use tz_localize() first to set the original timezone."
        )
    
    return df.tz_convert(UTC)


def to_local(df):
    """
    Convert DataFrame index to local timezone (Asia/Colombo).
    
    Use this before saving final outputs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with timezone-aware DatetimeIndex
    
    Returns:
    --------
    pd.DataFrame with Asia/Colombo DatetimeIndex
    
    Raises:
    -------
    ValueError if index is not timezone-aware
    """
    if df.index.tz is None:
        raise ValueError(
            "DataFrame index is not timezone-aware. "
            "Use tz_localize() first to set the original timezone."
        )
    
    return df.tz_convert(LOCAL_TZ)


def localize_utc(df):
    """
    Localize naive timestamps as UTC.
    
    Use this for data that is known to be UTC but doesn't have
    timezone info (e.g., some CSV exports).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with naive (non-timezone-aware) DatetimeIndex
    
    Returns:
    --------
    pd.DataFrame with UTC DatetimeIndex
    """
    if df.index.tz is not None:
        # Already timezone-aware, just convert
        return df.tz_convert(UTC)
    
    return df.tz_localize(UTC)


def localize_local(df):
    """
    Localize naive timestamps as Asia/Colombo.
    
    Use this for data that is known to be in local time but doesn't
    have timezone info.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with naive (non-timezone-aware) DatetimeIndex
    
    Returns:
    --------
    pd.DataFrame with Asia/Colombo DatetimeIndex
    """
    if df.index.tz is not None:
        # Already timezone-aware, just convert
        return df.tz_convert(LOCAL_TZ)
    
    return df.tz_localize(LOCAL_TZ)


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def get_complete_days(df, min_hours_per_day=20):
    """
    Filter DataFrame to only include complete days.
    
    A day is considered complete if it has at least min_hours_per_day
    hourly records.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    min_hours_per_day : int
        Minimum number of hours required for a day to be considered complete
    
    Returns:
    --------
    pd.DataFrame filtered to complete days only
    """
    # Group by date and count records
    daily_counts = df.groupby(df.index.date).size()
    
    # Find complete days
    complete_dates = daily_counts[daily_counts >= min_hours_per_day].index
    
    # Filter original DataFrame
    return df[df.index.date.isin(complete_dates)]


def align_to_daily(df, agg_func="sum"):
    """
    Aggregate hourly data to daily, respecting timezone.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with hourly data and DatetimeIndex
    agg_func : str or callable
        Aggregation function ('sum', 'mean', etc.)
    
    Returns:
    --------
    pd.DataFrame with daily data
    """
    return df.resample("D").agg(agg_func)


# =====================================================================
# TESTING
# =====================================================================

if __name__ == "__main__":
    # Test the utilities
    print("="*60)
    print("TIMEZONE UTILITIES TEST")
    print("="*60)
    
    # Test loading Meteocontrol data (scripts/shared/ -> ../../data/)
    meteo_path = Path(__file__).parent.parent.parent / "data" / "meteocontrol_actual.csv"
    
    if meteo_path.exists():
        print(f"\nLoading: {meteo_path}")
        df = load_meteocontrol(meteo_path)
        
        print(f"\nLoaded {len(df)} records")
        print(f"Timezone: {df.index.tz}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        print("\nSample data (first 5 rows):")
        print(df[["day", "actual_kwh"]].head())
        
        # Test UTC conversion
        df_utc = to_utc(df)
        print(f"\nConverted to UTC: {df_utc.index.tz}")
        print(f"First timestamp (local): {df.index[0]}")
        print(f"First timestamp (UTC):   {df_utc.index[0]}")
        
        # Convert back to local
        df_local = to_local(df_utc)
        print(f"\nConverted back to local: {df_local.index.tz}")
        print(f"First timestamp: {df_local.index[0]}")
        
    else:
        print(f"Test file not found: {meteo_path}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

