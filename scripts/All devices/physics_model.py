#!/usr/bin/env python3
"""
================================================================================
UNIFIED PHYSICS-BASED PV GENERATION MODEL
================================================================================

Production-grade Solcast → PV Generation pipeline that reproduces PVsyst physics
and loss ordering as closely as possible using:

  - Perez POA transposition (with dni_extra for accuracy)
  - AOI + IAM from PVsyst
  - SAPM temperature matched to PVsyst Uc/Uv
  - Far-shading (when configured in plant_config.json)
  - Sequential DC loss chain (PVsyst order)
  - Plant-level inverter clipping (PVsyst methodology)
  - AC wiring losses
  - Correct PVsyst orientation weighting

USAGE:
------
  # Read from local CSV file (default)
  python physics_model.py --source csv

  # Read from custom CSV path
  python physics_model.py --source csv --csv-path ../data/custom.csv

  # Use pvlib clear sky models (Ineichen - most accurate)
  python physics_model.py --source pvlib_clearsky --start 20251201 --end 20251215 --clearsky-model ineichen

  # Use Simplified Solis model
  python physics_model.py --source pvlib_clearsky --start 20251201 --end 20251215 --clearsky-model simplified_solis

  # Use Haurwitz model (fastest)
  python physics_model.py --source pvlib_clearsky --start 20251201 --end 20251215 --clearsky-model haurwitz

  # Show help
  python physics_model.py --help

NOTE:
-----
Exact numerical agreement with PVsyst requires validation against the
plant's PVsyst report and, where available, inverter efficiency curves.

================================================================================
"""

import argparse
import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

# =====================================================================
# CONFIGURATION LOADER
# =====================================================================

def load_config(config_path=None):
    """
    Load plant configuration from JSON file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
    
    Returns:
        dict: Plant configuration parameters
    """
    if config_path is None:
        # Path: scripts/python_physics/ -> ../../config/
        config_path = Path(__file__).parent.parent.parent / "config" / "plant_config.json"
    
    with open(config_path, "r") as f:
        return json.load(f)

# Load configuration
config = load_config()

# =====================================================================
# ⚠️ CRITICAL USER INPUTS — MUST BE VERIFIED PER PLANT
# =====================================================================
# These parameters are NOT generic defaults.
# They MUST be verified for EACH new solar plant.
#
# If ANY of the following values are incorrect or assumed without
# validation, the generation estimate will be physically inconsistent
# and may deviate from PVsyst by >10–30%.
#
# IMPORTANT:
# PVsyst is the REFERENCE model.
# This script is a PVsyst-consistent reimplementation, not an optimizer.
#
# Sources for each parameter are explicitly noted below.
#
# ---------------------------------------------------------------------
# 1) Solcast API + Location
# ---------------------------------------------------------------------
# API_KEY:
#   → Your own Solcast API key
#   → Obtained from Solcast dashboard (NOT from PVsyst)
#
# lat, lon:
#   → Exact plant latitude / longitude
#   → PVsyst Simulation Report → Page 2 → "Geographical Site"
#   → Must match the coordinates used in the PVsyst simulation
#
# period:
#   → Time resolution of Solcast data
#   → Independent of PVsyst internal timestep
#   → Choose based on modeling intent:
#       - "PT60M" for hourly energy modeling
#       - "PT30M" / "PT15M" only if sub-hourly dynamics are required
#
# TIMEZONE:
#   → Local plant timezone (used ONLY for labeling / reporting)
#   → PVsyst Simulation Report → Page 2 → "Time zone"
#   → ALL internal calculations are performed in UTC
#
# ---------------------------------------------------------------------
# 2) Array Orientations (HIGH-RISK INPUT)
# ---------------------------------------------------------------------
# orientations:
#   → MUST be extracted from PVsyst "Orientation" sections
#   → Each UNIQUE (tilt, azimuth) combination is a separate entry
#
# For EACH orientation:
#   - tilt (degrees):
#       PVsyst report → "General parameters" → Orientation #n
#
#   - azimuth (degrees):
#       PVsyst convention:
#         0 = South
#         +90 = West
#         -90 = East
#         ±180 = North
#
#   - module_count:
#       → Derived from plant layout / as-built drawings
#       → Count = strings × modules per string per roof/array plane
#       → MUST satisfy:
#           sum(module_count) == total modules reported by PVsyst
#
# ⚠ Do NOT guess module counts.
# ⚠ Do NOT evenly split unless documentation is unavailable
#    (and clearly mark as approximation).
#
# ---------------------------------------------------------------------
# 3) Module Geometry & Electrical Parameters
# ---------------------------------------------------------------------
# module_area (m² per module):
#   → PRIMARY source: Manufacturer datasheet / as-built drawings
#   → Secondary cross-check: PVsyst report → "Total PV power" → Module area
#   → PVsyst reports TOTAL module area, not per-module geometry
#
# module_efficiency_stc:
#   → PVsyst report → Loss Diagram
#       ("efficiency at STC = xx.xx% PV conversion")
#   → This value is already PVsyst-consistent and SHOULD be used directly
#   → NOTE: May differ slightly from datasheet (use PVsyst value for validation)
#
# gamma_p (power temperature coefficient):
#   → NOT explicitly printed in the PVsyst report
#   → Source MUST be one of:
#       - PV module datasheet, OR
#       - PVsyst internal module database
#   → Typical crystalline silicon range:
#       -0.003 to -0.004 per °C
#
# ---------------------------------------------------------------------
# 4) Inverter & System-Level Parameters
# ---------------------------------------------------------------------
# INV_AC_RATING_KW:
#   → PVsyst report → Page 2 → "System summary"
#   → Use TOTAL inverter AC power (sum of all inverters): Pnom total
#   → Value in kW for direct use with kW-based power calculations
#
# PDC_THRESHOLD_KW:
#   → Inverter startup / minimum operating power (Pmin / Pthresh)
#   → NOT numerically exposed in PVsyst simulation reports
#   → Obtain from:
#       - PVsyst report → Loss Diagram → "Power threshold loss"
#       - If reported as 0.0%, set PDC_THRESHOLD_KW = 0.0
#       - Otherwise obtain from PVsyst inverter component (.OND), OR
#       - Manufacturer datasheet
#
# ---------------------------------------------------------------------
# 5) Optical & Shading Parameters
# ---------------------------------------------------------------------
# albedo:
#   → PVsyst report → Page 2 → Project settings → Albedo
#
# far_shading:
#   → PVsyst report → Loss Diagram → "Far shading" or "Horizon" loss
#   → Value = 1 - (loss % / 100)
#   → Set to 1.0 if no shading is defined in PVsyst
#   → Applied multiplicatively to POA irradiance BEFORE IAM
#
# NOTE ON SHADING / HORIZON LOSSES:
# Apply shading factor IF PVsyst report shows:
# - Near shading loss (from shading scene)
# - Far shading loss (from horizon profile)
# - Explicit shading loss percentage in Loss Diagram
#
# Do NOT apply shading factor if:
# - PVsyst shows "free horizon"
# - No shading scene is defined
# - Only "Global incident in coll. plane" loss appears
#   (this is transposition geometry, already in Perez)
#
# ---------------------------------------------------------------------
# 6) DC-Side Loss Factors (PVsyst-derived)
# ---------------------------------------------------------------------
# soiling, LID, mismatch, dc_wiring, module_quality:
#   → PVsyst report → "Array losses" section / Loss Diagram
#   → These are ANNUAL AVERAGE losses
#   → Applied on the DC side only
#   → Order must remain:
#       Soiling → LID → Module quality → Mismatch → DC wiring
#   → module_quality is often a GAIN (negative loss), e.g., +0.8%
#   → Extract EXACT values from PVsyst loss diagram
#
# ---------------------------------------------------------------------
# 7) AC-Side Loss Factors (PVsyst-derived)
# ---------------------------------------------------------------------
# ac_wiring_loss:
#   → PVsyst report → Loss Diagram → "AC ohmic loss"
#   → Applied AFTER inverter conversion, BEFORE reporting
#   → Typically 0.2–0.5%
#
# ---------------------------------------------------------------------
# 8) IAM Table (MODULE-SPECIFIC)
# ---------------------------------------------------------------------
# iam_angles / iam_values:
#   → PVsyst report → "IAM loss factor" → User-defined profile
#   → Module-specific optical behavior
#   → MUST NOT be reused across different module models
#   → Interpolated against angle-of-incidence (AOI)
#
# ---------------------------------------------------------------------
# 9) Thermal Model Parameters
# ---------------------------------------------------------------------
# sapm_params:
#   → Chosen to approximate PVsyst thermal model (Uc / Uv)
#   → PVsyst report → "Thermal Loss factor"
#   → PVsyst Uc/Uv values guide SAPM mounting selection
#   → IMPORTANT: SAPM and PVsyst use different thermal equations
#   → Exact cell temperature match is NOT possible
#   → Validation is performed at ANNUAL energy/PR level, not hourly
#
# VALIDATION RULE:
#   If annual AC energy or PR deviates from PVsyst by >3–5%,
#   revisit thermal model selection FIRST.
#
# ---------------------------------------------------------------------
# FINAL WARNING
# ---------------------------------------------------------------------
# This script is NOT plug-and-play.
# It is a PVsyst-consistent forward model.
#
# ALWAYS validate against PVsyst:
#   - Annual POA irradiation
#   - Annual DC energy (EArray)
#   - Annual AC energy (E_Grid)
#   - Performance Ratio (PR)
#
# BEFORE using outputs for BI, forecasting, or operations.
# =====================================================================


# =====================================================================
# 1) PARAMETERS FROM CONFIG FILE
# =====================================================================
# All plant-specific parameters are loaded from config/plant_config.json
# Modify that file for different plants - do not hardcode values here.

# Location
lat = config["location"]["lat"]
lon = config["location"]["lon"]
TIMEZONE = config["location"]["timezone"]
SITE_ALTITUDE_M = config["location"]["altitude_m"]

# Orientations
orientations = config["orientations"]

# Module parameters
module_area = config["module"]["area_m2"]
total_module_area = sum(o["module_count"] * module_area for o in orientations)
module_efficiency_stc = config["module"]["efficiency_stc"]
gamma_p = config["module"]["gamma_p"]

# Inverter parameters
INV_AC_RATING_KW = config["inverter"]["ac_rating_kw"]
PDC_THRESHOLD_KW = config["inverter"]["dc_threshold_kw"]

# Losses
soiling = config["losses"]["soiling"]
LID = config["losses"]["lid"]
module_quality = config["losses"]["module_quality"]
mismatch = config["losses"]["mismatch"]
dc_wiring = config["losses"]["dc_wiring"]
ac_wiring_loss = config["losses"]["ac_wiring"]
albedo = config["losses"]["albedo"]
far_shading = config["losses"].get("far_shading", 1.0)  # Default to 1.0 (no shading)

# DC loss factor (computed from individual losses)
dc_loss_factor = (1 - soiling) * (1 - LID) * (1 - module_quality) * (1 - mismatch) * (1 - dc_wiring)

# IAM table
iam_angles = np.array(config["iam"]["angles"])
iam_values = np.array(config["iam"]["values"])

# Default weather values
DEFAULT_WIND_SPEED_MS = config["defaults"]["wind_speed_ms"]
DEFAULT_AIR_TEMP_C = config["defaults"]["air_temp_c"]

# Thermal model
sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][config["thermal_model"]]

# =====================================================================
# INVERTER EFFICIENCY MODEL (from config)
# =====================================================================
USE_INVERTER_CURVE = config["inverter"]["use_efficiency_curve"]
inverter_eff_curve_kw = np.array(config["inverter"]["efficiency_curve_kw"])
inverter_eff_curve_eta = np.array(config["inverter"]["efficiency_curve_eta"])
FLAT_EFFICIENCY = config["inverter"]["flat_efficiency"]

# Date range for Solcast data
start_date = config["date_range"]["start"]
end_date = config["date_range"]["end"]

def inverter_efficiency(pdc_kw):
    """
    Returns inverter efficiency η as a function of DC power (kW).
    Uses curve interpolation or flat efficiency based on config.
    """
    if USE_INVERTER_CURVE:
        return np.interp(np.asarray(pdc_kw), inverter_eff_curve_kw, inverter_eff_curve_eta)
    else:
        return FLAT_EFFICIENCY


# =====================================================================
# 2) PVLIB CLEAR SKY DATA GENERATION
# =====================================================================

def fetch_pvlib_clearsky(lat, lon, altitude, start_date_str, end_date_str, model='ineichen', freq='1h'):
    """
    Generate clear sky GHI, DNI, DHI data using pvlib clear sky models.
    
    Parameters:
    -----------
    lat, lon : float
        Location coordinates
    altitude : float
        Site altitude in meters
    start_date_str, end_date_str : str
        Date strings in format YYYYMMDD or ISO format
    model : str
        Clear sky model to use. Options:
        - 'ineichen' (default): Ineichen clear sky model (requires linke turbidity)
        - 'simplified_solis': Simplified Solis clear sky model
        - 'haurwitz': Simple Haurwitz model (DNI approximation)
    freq : str
        Time frequency (default '1h' for hourly)
    
    Returns:
    --------
    pd.DataFrame with columns: ghi, dni, dhi, air_temp (estimated), wind_speed (default)
    Index: UTC timestamps
    """
    print("\n" + "="*60)
    print("PVLIB CLEAR SKY MODEL")
    print("="*60)
    print(f"Model: {model}")
    print(f"Location: lat={lat}, lon={lon}, altitude={altitude}m")
    
    # Parse dates
    try:
        if len(start_date_str) == 8:  # YYYYMMDD format
            start_dt = pd.to_datetime(start_date_str, format='%Y%m%d')
            end_dt = pd.to_datetime(end_date_str, format='%Y%m%d')
        else:  # ISO format
            start_dt = pd.to_datetime(start_date_str)
            end_dt = pd.to_datetime(end_date_str)
    except:
        print(f"⚠️  Error parsing dates: {start_date_str}, {end_date_str}")
        return None
    
    # Create time index in UTC
    times = pd.date_range(start=start_dt, end=end_dt, freq=freq, tz='UTC')
    
    # Create location object
    location = Location(latitude=lat, longitude=lon, altitude=altitude)
    
    # Calculate solar position
    print("Calculating solar position...")
    solar_position = location.get_solarposition(times)
    
    # Calculate clear sky irradiance based on selected model
    print(f"Generating clear sky irradiance using {model} model...")
    
    if model == 'ineichen':
        # Ineichen clear sky model (most accurate, requires linke turbidity)
        clearsky = location.get_clearsky(times, model='ineichen')
    elif model == 'simplified_solis':
        # Simplified Solis clear sky model
        clearsky = location.get_clearsky(times, model='simplified_solis')
    elif model == 'haurwitz':
        # Simple Haurwitz model (only GHI, DNI/DHI approximated)
        clearsky = location.get_clearsky(times, model='haurwitz')
    else:
        print(f"⚠️  Unknown clear sky model: {model}")
        print("   Valid options: ineichen, simplified_solis, haurwitz")
        return None
    
    # Create DataFrame with expected column names
    # Haurwitz model only provides 'ghi', so we need to handle missing dni/dhi
    df_dict = {'period_end': times}
    
    if 'ghi' in clearsky.columns:
        df_dict['ghi'] = clearsky['ghi']
    if 'dni' in clearsky.columns:
        df_dict['dni'] = clearsky['dni']
    else:
        # Estimate DNI from GHI if not available (Haurwitz model)
        # Simple approximation: DNI ≈ GHI when sun is high
        zenith = solar_position['zenith'].values
        df_dict['dni'] = clearsky['ghi'] * np.cos(np.radians(zenith))
        df_dict['dni'] = np.maximum(df_dict['dni'], 0)
    
    if 'dhi' in clearsky.columns:
        df_dict['dhi'] = clearsky['dhi']
    else:
        # Estimate DHI from GHI and DNI if not available
        # DHI = GHI - DNI * cos(zenith)
        zenith = solar_position['zenith'].values
        df_dict['dhi'] = clearsky['ghi'] - df_dict['dni'] * np.cos(np.radians(zenith))
        df_dict['dhi'] = np.maximum(df_dict['dhi'], 0)
    
    df = pd.DataFrame(df_dict)
    
    # Estimate air temperature based on time of day and solar elevation
    # Simple model: base temp + solar heating effect
    hour = times.hour.values
    solar_elevation = solar_position['elevation'].values
    
    # Base temperature varies by time of year (simplified sinusoidal model)
    day_of_year = times.dayofyear.values
    base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Annual cycle
    
    # Daily temperature variation (warmer during day)
    daily_variation = 8 * np.sin(2 * np.pi * (hour - 6) / 24)
    daily_variation = np.maximum(daily_variation, 0)  # Only add heat during day
    
    # Solar heating effect (proportional to solar elevation)
    solar_heating = np.maximum(0, solar_elevation) * 0.15
    
    df['air_temp'] = base_temp + daily_variation + solar_heating
    
    # Default wind speed (constant, can be overridden)
    df['wind_speed'] = 2.0  # m/s - typical moderate wind
    
    df.set_index('period_end', inplace=True)
    
    print("✓ Clear sky data successfully generated")
    print(f"  GHI range: {df['ghi'].min():.1f} to {df['ghi'].max():.1f} W/m²")
    print(f"  DNI range: {df['dni'].min():.1f} to {df['dni'].max():.1f} W/m²")
    print(f"  DHI range: {df['dhi'].min():.1f} to {df['dhi'].max():.1f} W/m²")
    print(f"  Estimated temp range: {df['air_temp'].min():.1f}°C to {df['air_temp'].max():.1f}°C")
    print(f"  Records: {len(df)}")
    print("="*60 + "\n")
    
    return df


# =====================================================================
# 3) CSV DATA LOADING
# =====================================================================

def load_csv_data(csv_path):
    """
    Load irradiance data from local CSV file.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to CSV file containing irradiance data
    
    Returns:
    --------
    pd.DataFrame with DatetimeIndex (UTC) and columns: ghi, dni, dhi
    """
    print(f"Loading irradiance data from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Find timestamp column
    time_col_candidates = ["time", "timestamp", "period_end", "datetime"]
    
    for col in time_col_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
            df = df.set_index(col)
            break
    else:
        raise RuntimeError("No valid timestamp column found in CSV. Expected one of: " + 
                          ", ".join(time_col_candidates))
    
    # Sort & validate
    df = df.sort_index()
    required_cols = ["ghi", "dni", "dhi"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        raise RuntimeError(f"CSV missing required irradiance columns: {missing}")
    
    print(f"✓ Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
    
    return df


# =====================================================================
# 5) WEATHER DATA PROCESSING
# =====================================================================

def ensure_weather_data(df):
    """
    Ensure weather data (air_temp, wind_speed) is available.
    Always uses fixed default values (independent from external data sources).
    """
    # Fixed default values (optimized for typical conditions)
    DEFAULT_AIR_TEMP = 25.0  # °C - typical ambient temperature
    DEFAULT_WIND_SPEED = 1.0  # m/s - low wind condition
    
    print("\n📋 Weather data (using fixed defaults, no external files):")
    df["air_temp"] = DEFAULT_AIR_TEMP
    print(f"   - air_temp: {DEFAULT_AIR_TEMP}°C (fixed default)")
    df["wind_speed"] = DEFAULT_WIND_SPEED
    print(f"   - wind_speed: {DEFAULT_WIND_SPEED} m/s (fixed default)")
    print("✓ Weather data configured")
    
    return df


# =====================================================================
# 6) PV MODEL (Perez POA → IAM → SAPM → DC → AC → Clipping)
# =====================================================================

def compute_pv_ac(df_weather):
    """
    Compute AC power output from weather data using physics-based model.
    
    Parameters:
    -----------
    df_weather : pd.DataFrame
        DataFrame with DatetimeIndex (UTC) and columns: ghi, dni, dhi, air_temp, wind_speed
    
    Returns:
    --------
    pd.Series with AC power output in kW
    """
    # Create site location
    site = Location(
        lat,
        lon,
        altitude=SITE_ALTITUDE_M if SITE_ALTITUDE_M is not None else 0
    )
    solpos = site.get_solarposition(df_weather.index)
    
    plant_ac = pd.Series(0.0, index=df_weather.index)
    
    # Extraterrestrial radiation for Perez model accuracy
    dni_extra = pvlib.irradiance.get_extra_radiation(df_weather.index)
    
    for o in orientations:
        tilt = o["tilt"]
        azimuth = o["azimuth"]
        area_fraction = (o["module_count"] * module_area) / total_module_area
        
        # Perez POA transposition (GHI/DNI/DHI → POA)
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
        
        # Apply far-shading BEFORE IAM (if configured)
        # far_shading = 1.0 means no shading, < 1.0 means shading loss
        if far_shading < 1.0:
            poa = poa * far_shading
        
        # AOI + IAM (optical correction)
        aoi = pvlib.irradiance.aoi(
            tilt, azimuth,
            solpos["zenith"], solpos["azimuth"]
        )
        iam = np.interp(aoi, iam_angles, iam_values)
        poa_optical = poa * iam
        
        # SAPM cell temperature (uses POA BEFORE IAM, per PVsyst methodology)
        # CRITICAL: Do NOT use poa_optical here; thermal losses depend on
        # total absorbed irradiance, not optically corrected irradiance
        cell_temp = pvlib.temperature.sapm_cell(
            poa,  # POA before IAM (correct per methodology)
            df_weather["air_temp"],
            df_weather["wind_speed"],
            **sapm_params
        )
        
        # DC power per m² (kW/m²)
        pdc_kwm2 = poa_optical * module_efficiency_stc / 1000
        
        # Apply temperature coefficient
        pdc_kwm2_temp = pdc_kwm2 * (1 + gamma_p * (cell_temp - 25))
        
        # Apply DC losses (PVsyst order: soiling → LID → module quality → mismatch → DC wiring)
        pdc_kwm2_eff = pdc_kwm2_temp * dc_loss_factor
        
        # Scale to orientation's total area
        area_i = total_module_area * area_fraction
        pdc_total_kw = pdc_kwm2_eff * area_i
        
        # Apply inverter DC threshold (Pmin / Pthresh)
        if PDC_THRESHOLD_KW > 0:
            pdc_total_kw = pdc_total_kw.where(pdc_total_kw >= PDC_THRESHOLD_KW, 0.0)
        
        # Convert DC → AC using inverter efficiency
        pac_kw = pdc_total_kw * inverter_efficiency(pdc_total_kw)
        
        # Accumulate per-orientation AC power (clipping happens later)
        plant_ac += pac_kw
    
    # Plant-level inverter clipping (AFTER orientation aggregation, per PVsyst)
    # CRITICAL: Do NOT clip per-orientation; PVsyst clips at plant level
    plant_ac = plant_ac.clip(upper=INV_AC_RATING_KW)
    
    # Apply AC wiring losses (AFTER inverter, BEFORE reporting)
    plant_ac = plant_ac * (1 - ac_wiring_loss)
    
    return plant_ac


# =====================================================================
# 6b) SIMPLIFIED DAILY MODEL (for NASA POWER daily data)
# =====================================================================

def calculate_daily_energy_simple(df_daily):
    """
    Simplified daily energy model using daily irradiance totals.
    
    This is a simplified approach for quick estimates when only daily
    irradiance data is available (e.g., NASA POWER daily endpoint).
    
    Less accurate than hourly model because:
    - No hour-by-hour Perez POA transposition
    - Uses average temperature for entire day
    - No sub-daily inverter clipping effects
    
    Parameters:
    -----------
    df_daily : pd.DataFrame
        Daily data with columns:
        - ghi_kwh_m2_day: Daily GHI in kWh/m²/day
        - dhi_kwh_m2_day: Daily DHI in kWh/m²/day
        - dni_kwh_m2_day: Daily DNI in kWh/m²/day
        - temp_c_daily: Daily average temperature in °C
        - wind_mps_daily: Daily average wind speed in m/s
    
    Returns:
    --------
    pd.Series with daily AC energy output in kWh
    """
    print("\n" + "="*60)
    print("SIMPLIFIED DAILY MODEL")
    print("="*60)
    
    # System parameters from config
    total_module_area = sum(o["module_count"] for o in orientations) * module_area
    
    # Average POA factor (ratio of POA irradiance to GHI for tilted arrays)
    # This is a simplification - actual value depends on tilt, azimuth, and season
    # For tropical locations with multiple orientations, 1.0-1.1 is typical
    avg_poa_factor = 1.05
    
    # System efficiency (simplified)
    # Includes: module efficiency, temperature losses, DC losses, inverter losses, AC losses
    temp_ref = 25.0
    temp_coeff = config["module"]["gamma_p"]  # Typically -0.0034
    
    # Daily losses (product of individual loss factors)
    loss_factors = [
        1 - config["losses"]["soiling"],
        1 - config["losses"]["lid"],
        1 + config["losses"]["module_quality"],  # Can be positive (quality gain)
        1 - config["losses"]["mismatch"],
        1 - config["losses"]["dc_wiring"],
        1 - config["losses"]["ac_wiring"],
        config["losses"]["far_shading"],
    ]
    dc_loss_factor = 1.0
    for f in loss_factors:
        dc_loss_factor *= f
    
    # Module STC efficiency
    module_eff = config["module"]["efficiency_stc"]
    
    # Inverter efficiency (simplified flat value)
    inv_eff = config["inverter"]["flat_efficiency"]
    
    # Calculate daily energy for each day
    daily_energy = []
    
    for idx, row in df_daily.iterrows():
        # Get irradiance (kWh/m²/day)
        ghi = row.get("ghi_kwh_m2_day", 0)
        
        # Skip invalid days
        if pd.isna(ghi) or ghi <= 0:
            daily_energy.append(np.nan)
            continue
        
        # Estimate POA irradiance (kWh/m²/day)
        poa_daily = ghi * avg_poa_factor
        
        # Temperature correction
        temp = row.get("temp_c_daily", temp_ref)
        if pd.isna(temp):
            temp = temp_ref
        temp_factor = 1 + temp_coeff * (temp - temp_ref)
        
        # DC energy before losses (kWh)
        dc_energy = poa_daily * total_module_area * module_eff * temp_factor
        
        # Apply DC losses
        dc_energy_net = dc_energy * dc_loss_factor
        
        # Apply inverter efficiency
        ac_energy = dc_energy_net * inv_eff
        
        # Apply inverter clipping (daily energy cap = 24h * AC rating)
        max_daily_energy = INV_AC_RATING_KW * 24
        ac_energy = min(ac_energy, max_daily_energy)
        
        daily_energy.append(ac_energy)
    
    result = pd.Series(daily_energy, index=df_daily.index, name="energy_kWh")
    
    print(f"Total days: {len(df_daily)}")
    print(f"Valid days: {result.notna().sum()}")
    print(f"Total energy: {result.sum():.1f} kWh")
    print(f"Average daily: {result.mean():.1f} kWh")
    print("="*60)
    
    return result


# =====================================================================
# 7) MAIN EXECUTION
# =====================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Physics-based PV generation model (No external APIs required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python physics_model.py --source csv
  python physics_model.py --source csv --csv-path ../data/custom.csv
  python physics_model.py --source pvlib_clearsky --start 20260110 --end 20260115
  python physics_model.py --source pvlib_clearsky --start 20260110 --end 20260115 --clearsky-model ineichen
        """
    )
    
    parser.add_argument(
        "--source",
        choices=["csv", "pvlib_clearsky"],
        default="pvlib_clearsky",
        help="Data source: 'csv' for local file, 'pvlib_clearsky' for clear sky models (default: pvlib_clearsky)"
    )
    
    parser.add_argument(
        "--csv-path",
        default=str(Path(__file__).parent.parent.parent.parent / "data" / "solcast_irradiance.csv"),
        help="Path to CSV file when using --source csv"
    )
    
    parser.add_argument(
        "--start",
        default="20260101",
        help="Start date in YYYYMMDD format (for pvlib_clearsky source, default: 20260101)"
    )
    
    parser.add_argument(
        "--end",
        default="20260112",
        help="End date in YYYYMMDD format (for pvlib_clearsky source, default: 20260112)"
    )
    
    parser.add_argument(
        "--clearsky-model",
        type=str,
        default="ineichen",
        choices=["ineichen", "simplified_solis", "haurwitz"],
        help="Clear sky model for pvlib_clearsky source (default: ineichen)"
    )
    
    parser.add_argument(
        "--frequency",
        type=str,
        default="1h",
        help="Time frequency for pvlib_clearsky data (default: 1h for hourly)"
    )
    
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent.parent / "output" / "pv_generation.csv"),
        help="Output CSV path"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PHYSICS-BASED PV GENERATION MODEL")
    print("="*60)
    print(f"Data source: {args.source}")
    if args.source == "csv":
        print(f"CSV path: {args.csv_path}")
    elif args.source == "pvlib_clearsky":
        print(f"pvlib clear sky model: {args.clearsky_model}")
        print(f"Date range: {args.start} to {args.end}")
        print(f"Frequency: {args.frequency}")
    print(f"Far-shading factor: {far_shading}")
    print("="*60 + "\n")
    
    # =====================================================================
    # LOAD DATA
    # =====================================================================
    
    if args.source == "pvlib_clearsky":
        # Generate clear sky data using pvlib models
        if not args.start or not args.end:
            print("⚠️ ERROR: --start and --end required for pvlib_clearsky source")
            print("   Example: --start 20251201 --end 20251215")
            sys.exit(1)
        
        df = fetch_pvlib_clearsky(
            lat, lon, SITE_ALTITUDE_M, 
            args.start, args.end,
            model=args.clearsky_model,
            freq=args.frequency
        )
        
        if df is None:
            print("⚠️ ERROR: Failed to generate clear sky data")
            sys.exit(1)
        
    else:
        # Load from CSV
        df = load_csv_data(args.csv_path)
    
    # =====================================================================
    # ENSURE WEATHER DATA
    # =====================================================================
    
    df = ensure_weather_data(df)
    
    # =====================================================================
    # TIMESTAMP VALIDATION
    # =====================================================================
    
    # Ensure time index is strictly increasing
    if not df.index.is_monotonic_increasing:
        print("⚠️  Timestamp order issue detected – sorting index")
        df = df.sort_index()
    
    # Enforce regular hourly grid if source data drifts
    EXPECTED_FREQ = "h"
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq != EXPECTED_FREQ and inferred_freq != "H":
        print(f"⚠️  Irregular timestep detected (inferred: {inferred_freq})")
        print("   Resampling to nearest hourly timestamps")
        df = df.resample(EXPECTED_FREQ).nearest()
    
    # =====================================================================
    # COMPUTE PV OUTPUT
    # =====================================================================
    
    print("\nComputing PV AC output...")
    plant_ac = compute_pv_ac(df)
    
    # Compute timestep duration for energy integration
    dt_hours = (df.index[1] - df.index[0]).total_seconds() / 3600
    
    # Convert to local timezone for reporting/output only
    plant_ac_local = plant_ac.tz_convert(TIMEZONE)
    
    # Integrate power to energy (kWh)
    energy = (plant_ac_local * dt_hours).sum()
    
    # =====================================================================
    # SAVE OUTPUT
    # =====================================================================
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    out = pd.DataFrame({"AC_kW": plant_ac_local})
    out.to_csv(output_path)
    
    # =====================================================================
    # RESULTS SUMMARY
    # =====================================================================
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Output saved: {output_path}")
    print(f"Total energy (kWh): {energy:.1f}")
    print(f"Period: {df.index.min()} to {df.index.max()}")
    print(f"Data source: {args.source}")
    print("="*60)
    
    print("\nSample output (first 10 rows):")
    print(out.head(10))
    
    print("\n" + "="*60)
    print("VALIDATION CHECKLIST (compare with PVsyst report)")
    print("="*60)
    print("□ Annual POA irradiation (kWh/m²)")
    print("□ Annual DC energy (EArray)")
    print("□ Annual AC energy (E_Grid)")
    print("□ Performance Ratio (PR)")
    print("□ Acceptable deviation: ±2-5%")
    print("="*60)
    
    return plant_ac


if __name__ == "__main__":
    main()
