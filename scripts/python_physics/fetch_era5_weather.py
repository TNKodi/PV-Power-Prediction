"""
FINAL: ERA5 Weather Fetcher for PV Validation
Purpose:
- Provide physically consistent air temperature & wind speed
- For hindcast validation against actual PV generation
- Designed to plug directly into PVsyst-consistent PV models

Outputs:
- era5_weather.csv
  Columns:
    - time (UTC)
    - air_temp (°C)
    - wind_speed (m/s)

Authoritative source:
- ECMWF ERA5 reanalysis (Copernicus CDS)
"""

import cdsapi
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, UTC

# =====================================================================
# USER INPUTS
# =====================================================================

LAT = 8.342368984714714
LON = 80.37623529556957

# Past-week hindcast window (UTC)

END_DATE = datetime.now(UTC).date()
START_DATE = END_DATE - timedelta(days=7)

# Path: scripts/python_physics/ -> ../../output, ../../data
OUTPUT_NETCDF = r"..\..\output\era5_weather.nc"
OUTPUT_CSV = r"..\..\data\era5_weather.csv"

# =====================================================================
# ERA5 DOWNLOAD
# =====================================================================

print("Requesting ERA5 weather data...")
print(f"Period: {START_DATE} → {END_DATE}")
print(f"Location: lat={LAT}, lon={LON}")

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": str(START_DATE.year),
        "month": f"{START_DATE.month:02d}",
        "day": [
            f"{d:02d}"
            for d in range(START_DATE.day, END_DATE.day + 1)
        ],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [
            LAT + 0.1,  # North
            LON - 0.1,  # West
            LAT - 0.1,  # South
            LON + 0.1,  # East
        ],
        "format": "netcdf",
    },
    OUTPUT_NETCDF,
)

# =====================================================================
# PROCESS NETCDF → PV-READY WEATHER
# =====================================================================

print("Processing ERA5 data...")

ds = xr.open_dataset(OUTPUT_NETCDF)

df = ds.to_dataframe().reset_index()

# ---------------------------------------------------------------------
# DETECT TIME COLUMN (ERA5-SAFE)
# ---------------------------------------------------------------------
if "time" in df.columns:
    time_col = "time"
elif "valid_time" in df.columns:
    time_col = "valid_time"
else:
    raise RuntimeError(f"Cannot find time column in ERA5 data. Columns: {df.columns}")

# ---------------------------------------------------------------------
# SPATIAL AVERAGING (lat/lon → point)
# ---------------------------------------------------------------------
df = (
    df
    .groupby(time_col, as_index=False)
    .mean(numeric_only=True)
)

# ---------------------------------------------------------------------
# UNIT CONVERSIONS
# ---------------------------------------------------------------------
# Temperature: Kelvin → Celsius
df["air_temp"] = df["t2m"] - 273.15

# Wind speed magnitude
df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)

# ---------------------------------------------------------------------
# FINAL WEATHER DF
# ---------------------------------------------------------------------
weather_df = (
    df[[time_col, "air_temp", "wind_speed"]]
    .rename(columns={time_col: "time"})
    .set_index("time")
    .sort_index()
)

weather_df.index = pd.to_datetime(weather_df.index, utc=True)

# =====================================================================
# SAVE CSV
# =====================================================================

weather_df.to_csv(OUTPUT_CSV)

print("ERA5 weather file created successfully.")
print(f"Saved as: {OUTPUT_CSV}")
print("\nSample:")
print(weather_df.head(10))

# =====================================================================
# VALIDATION NOTES
# =====================================================================
print("\nNOTES:")
print("- Data is UTC (no timezone conversion applied)")
print("- air_temp in °C, wind_speed in m/s")
print("- Suitable for SAPM / PVsyst-consistent thermal modeling")
print("- Spatial averaging applied (recommended for ERA5 point use)")
