import requests
import time
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


# =========================================================
# CONFIGURATION
# =========================================================

TB_URL = "https://windforce.thingsnode.cc"
USERNAME = ""
PASSWORD = ""

ROOT_ASSET_ID = "78fda490-e08b-11f0-b68f-8f33a9d74e0c"   # Level 0 asset
TARGET_LEVEL = 3                   # We want Level-4 assets
START_DATE = "2026-01-14"
END_DATE = "2026-01-31"
TZ_LOCAL = "Asia/Colombo"  # All devices are in Sri Lanka

# =========================================================
# AUTHENTICATION
def tb_login():
    url = f"{TB_URL}/api/auth/login"
    payload = {"username": USERNAME, "password": PASSWORD}
    r = requests.post(url, json=payload)
    r.raise_for_status()
    token = r.json()["token"]
    return {"X-Authorization": f"Bearer {token}"}
HEADERS = tb_login()
print("✅ Logged into ThingsBoard")

# =========================================================

def get_device_name(device_id):
    """Get device name from device ID"""
    url = f"{TB_URL}/api/device/{device_id}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json().get("name", device_id)


def send_telemetry(device_id, telemetry_data):
    """Send telemetry data to ThingsBoard device
    
    Args:
        device_id: ThingsBoard device ID
        telemetry_data: Dictionary with telemetry key-value pairs
                       OR list of dicts with 'ts' and 'values' keys
    
    Returns:
        Response object from the API
    """
    url = f"{TB_URL}/api/plugins/telemetry/DEVICE/{device_id}/timeseries/ANY"
    r = requests.post(url, headers=HEADERS, json=telemetry_data)
    r.raise_for_status()
    return r


def get_devices_from_asset(asset_id):
    """Get DEVICE relations under an asset"""
    url = f"{TB_URL}/api/relations/info"
    params = {"fromId": asset_id, "fromType": "ASSET"}
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()

    devices = []
    for rel in r.json():
        if rel["to"]["entityType"] == "DEVICE":
            devices.append(rel["to"]["id"])
    return devices

def get_asset_children(asset_id):
    url=f"{TB_URL}/api/relations/info"
    params={"fromId": asset_id, "fromType": "ASSET"}
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()

    children = []
    for rel in r.json():
        if rel["to"]["entityType"] == "ASSET" and rel["type"] == "Contains":
            children.append(rel["to"]["id"])
    return children

def get_assets_at_level(root_asset_id, level):
    current_assets = [root_asset_id]

    for _ in range(1, level+1):
        next_assets = []
        for aid in current_assets:
            next_assets.extend(get_asset_children(aid))
        current_assets = next_assets
        print(f"✅ Retrieved assets at level {_}: {len(current_assets)} assets found")
        if not current_assets:
            print("⚠️ No more assets found at this level. Exiting.")
            break
    return current_assets


# =========================================================
# HELPER FUNCTIONS FOR SOLAR CALCULATION
# =========================================================

def inverter_efficiency(pdc_kw, device_config):
    """
    Returns inverter efficiency η as a function of DC power (kW).
    Uses curve interpolation or flat efficiency based on config.
    """
    if device_config.use_efficiency_curve:
        return np.interp(np.asarray(pdc_kw), device_config.efficiency_curve_kw, device_config.efficiency_curve_eta)
    else:
        return device_config.flat_efficiency


# =====================================================================
# 2) PVLIB CLEAR SKY DATA GENERATION
# =====================================================================

def fetch_pvlib_clearsky(lat, lon, altitude, start_date_str, end_date_str, model='ineichen', freq='1h', tz=TZ_LOCAL):
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
    
    # Parse dates and align to local timezone boundaries, then convert to UTC
    try:
        if len(start_date_str) == 8:  # YYYYMMDD format
            start_dt = pd.to_datetime(start_date_str, format='%Y%m%d')
            end_dt = pd.to_datetime(end_date_str, format='%Y%m%d')
        else:  # ISO format
            start_dt = pd.to_datetime(start_date_str)
            end_dt = pd.to_datetime(end_date_str)

        # Localize to provided timezone (e.g., Asia/Colombo) so day boundaries are local, then convert to UTC
        if tz:
            start_dt = start_dt.tz_localize(tz).tz_convert('UTC')
            end_dt = end_dt.tz_localize(tz).tz_convert('UTC')
    except Exception as e:
        print(f"⚠️  Error parsing dates: {start_date_str}, {end_date_str} ({e})")
        return None
    
    # Create time index in UTC (aligned to local day boundaries)
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

def compute_pv_ac(df_weather, device_config):
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
        device_config.lat,
        device_config.lon,
        altitude=device_config.altitude_m if device_config.altitude_m is not None else 0
    )
    solpos = site.get_solarposition(df_weather.index)
    
    plant_ac = pd.Series(0.0, index=df_weather.index)
    
    # Extraterrestrial radiation for Perez model accuracy
    dni_extra = pvlib.irradiance.get_extra_radiation(df_weather.index)
    
    for o in device_config.orientations:
        tilt = o["tilt"]
        azimuth = o["azimuth"]
        area_fraction = (o["module_count"] * device_config.module_area_m2) / device_config.total_module_area
        
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
            albedo=device_config.albedo
        )
        
        poa = irr["poa_global"].clip(lower=0)
        
        # Apply far-shading BEFORE IAM (if configured)
        # far_shading = 1.0 means no shading, < 1.0 means shading loss
        if device_config.far_shading < 1.0:
            poa = poa * device_config.far_shading
        
        # AOI + IAM (optical correction)
        aoi = pvlib.irradiance.aoi(
            tilt, azimuth,
            solpos["zenith"], solpos["azimuth"]
        )
        iam = np.interp(aoi, device_config.iam_angles, device_config.iam_values)
        poa_optical = poa * iam
        
        # SAPM cell temperature (uses POA BEFORE IAM, per PVsyst methodology)
        # CRITICAL: Do NOT use poa_optical here; thermal losses depend on
        # total absorbed irradiance, not optically corrected irradiance
        cell_temp = pvlib.temperature.sapm_cell(
            poa,  # POA before IAM (correct per methodology)
            df_weather["air_temp"],
            df_weather["wind_speed"],
            **device_config.sapm_params
        )
        
        # DC power per m² (kW/m²)
        pdc_kwm2 = poa_optical * device_config.module_efficiency_stc / 1000
        
        # Apply temperature coefficient
        pdc_kwm2_temp = pdc_kwm2 * (1 + device_config.gamma_p * (cell_temp - 25))
        
        # Apply DC losses (PVsyst order: soiling → LID → module quality → mismatch → DC wiring)
        pdc_kwm2_eff = pdc_kwm2_temp * device_config.dc_loss_factor
        
        # Scale to orientation's total area
        area_i = device_config.total_module_area * area_fraction
        pdc_total_kw = pdc_kwm2_eff * area_i
        
        # Apply inverter DC threshold (Pmin / Pthresh)
        if device_config.inv_dc_threshold_kw > 0:
            pdc_total_kw = pdc_total_kw.where(pdc_total_kw >= device_config.inv_dc_threshold_kw, 0.0)
        
        # Convert DC → AC using inverter efficiency
        pac_kw = pdc_total_kw * inverter_efficiency(pdc_total_kw, device_config)
        
        # Accumulate per-orientation AC power (clipping happens later)
        plant_ac += pac_kw
    
    # Plant-level inverter clipping (AFTER orientation aggregation, per PVsyst)
    # CRITICAL: Do NOT clip per-orientation; PVsyst clips at plant level
    plant_ac = plant_ac.clip(upper=device_config.inv_ac_rating_kw)
    
    # Apply AC wiring losses (AFTER inverter, BEFORE reporting)
    plant_ac = plant_ac * (1 - device_config.ac_wiring)
    
    return plant_ac


# =====================================================================
# 6b) SIMPLIFIED DAILY MODEL (for NASA POWER daily data)
# =====================================================================

def calculate_daily_energy_simple(df_daily, device_config):
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
    
    # System parameters from device config
    total_module_area = device_config.total_module_area
    
    # Average POA factor (ratio of POA irradiance to GHI for tilted arrays)
    # This is a simplification - actual value depends on tilt, azimuth, and season
    # For tropical locations with multiple orientations, 1.0-1.1 is typical
    avg_poa_factor = 1.05
    
    # System efficiency (simplified)
    # Includes: module efficiency, temperature losses, DC losses, inverter losses, AC losses
    temp_ref = 25.0
    temp_coeff = device_config.gamma_p
    
    # Daily losses (product of individual loss factors)
    loss_factors = [
        1 - device_config.soiling,
        1 - device_config.lid,
        1 + device_config.module_quality,  # Can be positive (quality gain)
        1 - device_config.mismatch,
        1 - device_config.dc_wiring,
        1 - device_config.ac_wiring,
        device_config.far_shading,
    ]
    dc_loss_factor = 1.0
    for f in loss_factors:
        dc_loss_factor *= f
    
    # Module STC efficiency
    module_eff = device_config.module_efficiency_stc
    
    # Inverter efficiency (simplified flat value)
    inv_eff = device_config.flat_efficiency
    
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
        max_daily_energy = device_config.inv_ac_rating_kw * 24
        ac_energy = min(ac_energy, max_daily_energy)
        
        daily_energy.append(ac_energy)
    
    result = pd.Series(daily_energy, index=df_daily.index, name="energy_kWh")
    
    print(f"Total days: {len(df_daily)}")
    print(f"Valid days: {result.notna().sum()}")
    print(f"Total energy: {result.sum():.1f} kWh")
    print(f"Average daily: {result.mean():.1f} kWh")
    print("="*60)
    
    return result


def aggregate_to_daily(hourly_power_series, dt_hours):
    """
    Aggregate hourly AC power to daily energy totals.
    
    Parameters:
    -----------
    hourly_power_series : pd.Series
        Hourly AC power in kW (with DatetimeIndex)
    dt_hours : float
        Timestep duration in hours (for energy integration)
    
    Returns:
    --------
    pd.DataFrame with daily energy (kWh) and average power (kW)
    """
    # Convert power (kW) to energy (kWh) for each hour
    hourly_energy = hourly_power_series * dt_hours
    
    # Aggregate to daily totals
    daily_energy = hourly_energy.resample('D').sum()
    daily_avg_power = hourly_power_series.resample('D').mean()
    daily_peak_power = hourly_power_series.resample('D').max()
    
    # Create DataFrame with daily statistics
    daily_df = pd.DataFrame({
        'energy_kwh': daily_energy,
        'avg_power_kw': daily_avg_power,
        'peak_power_kw': daily_peak_power
    })
    
    return daily_df


#========================================================
# Configuration Management Class
#========================================================

class DeviceConfigManager:
    """
    Manages all device configuration variables from JSON config file.
    Loads 50+ variables from config and provides clean, organized access.
    """
    
    def __init__(self, device_id,device_name):
        """Load and initialize config for a specific device."""
        config_path = Path("config") / f"{device_id}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.raw_config = json.load(f)
        
        self.device_id = device_id
        self.device_name=device_name
        self._load_all_parameters()
    
    def _load_all_parameters(self):
        """Extract all parameters from config JSON into instance variables."""
        
        # ===== LOCATION PARAMETERS =====
        self.lat = self.raw_config["location"]["lat"]
        self.lon = self.raw_config["location"]["lon"]
        self.altitude_m = self.raw_config["location"]["altitude_m"]
        self.timezone = self.raw_config["location"]["timezone"]
        
        # ===== ORIENTATION PARAMETERS =====
        self.orientations = self.raw_config["orientations"]
        
        # ===== MODULE PARAMETERS =====
        self.module_area_m2 = self.raw_config["module"]["area_m2"]
        self.total_module_area = sum(o["module_count"] * self.module_area_m2 for o in self.orientations)  
        self.module_efficiency_stc = self.raw_config["module"]["efficiency_stc"]
        self.gamma_p = self.raw_config["module"]["gamma_p"]

        
        # ===== INVERTER PARAMETERS =====
        self.inv_ac_rating_kw = self.raw_config["inverter"]["ac_rating_kw"]
        self.inv_dc_threshold_kw = self.raw_config["inverter"]["dc_threshold_kw"]
        self.use_efficiency_curve = self.raw_config["inverter"]["use_efficiency_curve"]
        self.efficiency_curve_kw = np.array(
            self.raw_config["inverter"]["efficiency_curve_kw"]
        )
        self.efficiency_curve_eta = np.array(
            self.raw_config["inverter"]["efficiency_curve_eta"]
        )
        self.flat_efficiency = self.raw_config["inverter"]["flat_efficiency"]
        
        # ===== LOSS PARAMETERS =====
        self.soiling = self.raw_config["losses"]["soiling"]
        self.lid = self.raw_config["losses"]["lid"]
        self.module_quality = self.raw_config["losses"]["module_quality"]
        self.mismatch = self.raw_config["losses"]["mismatch"]
        self.dc_wiring = self.raw_config["losses"]["dc_wiring"]
        self.ac_wiring = self.raw_config["losses"]["ac_wiring"]
        self.albedo = self.raw_config["losses"]["albedo"]
        self.far_shading = self.raw_config["losses"].get("far_shading", 1.0)
        
        # Compute DC loss factor
        self.dc_loss_factor = (
            (1 - self.soiling) * 
            (1 - self.lid) * 
            (1 - self.module_quality) * 
            (1 - self.mismatch) * 
            (1 - self.dc_wiring)
        )
        
        # ===== IAM PARAMETERS =====
        self.iam_angles = np.array(self.raw_config["iam"]["angles"])
        self.iam_values = np.array(self.raw_config["iam"]["values"])
        
        # ===== THERMAL MODEL PARAMETERS =====
        self.thermal_model = self.raw_config["thermal_model"]
        self.sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
            self.thermal_model
        ]
        
        # ===== DEFAULT WEATHER PARAMETERS =====
        self.wind_speed_ms = self.raw_config["defaults"]["wind_speed_ms"]
        self.air_temp_c = self.raw_config["defaults"]["air_temp_c"]
        
        # ===== DATE RANGE PARAMETERS =====
        self.start_date = self.raw_config["date_range"]["start"]
        self.end_date = self.raw_config["date_range"]["end"]
        
        # ===== API PARAMETERS (if exists) =====
        self.api_key = self.raw_config.get("api", {}).get("solcast_key", "")
        self.api_period = self.raw_config.get("api", {}).get("period", "PT60M")
    
    def get_summary(self):
        """Return a summary of loaded configuration."""
        return {
            "device_name": self.device_name,
            "location": f"({self.lat}, {self.lon})",
            "altitude": f"{self.altitude_m}m",
            "timezone": self.timezone,
            "total_modules": sum(o["module_count"] for o in self.orientations),
            "total_area": f"{self.total_module_area:.1f}m²",
            "inv_rating": f"{self.inv_ac_rating_kw}kW",
            "num_orientations": len(self.orientations),
        }
    
    def print_summary(self):
        """Print configuration summary."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print(f"DEVICE CONFIG: {self.device_id}")
        print("="*60)
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("="*60)


#========================================================
# Solar Power Calculation Functions
#========================================================

def write_device_telemetry(device_id, telemetry):
    url = f"{TB_URL}/api/plugins/telemetry/DEVICE/{device_id}/timeseries/ANY"
    r = requests.post(url, headers=HEADERS, json=telemetry)
    r.raise_for_status()



def Solor_Power_Calculation(device_details):




    """Calculate solar power for a device using its config."""
    device_id, device_name = device_details
    
    # Load configuration using the new class
    Device = DeviceConfigManager(device_id, device_name)
    Device.print_summary()
    
    # Now you can use: Device.lat, Device.lon, Device.total_module_area, etc.
    # All 50+ variables are organized and accessible
    
    print(f"\n✅ Calculating solar power for: {device_name}")
    print(f"   Device ID: {device_id}")
    
    # Your solar power calculation logic here
    # Example:
    # site = Location(config.lat, config.lon, altitude=config.altitude_m)

    df=fetch_pvlib_clearsky(
        Device.lat,
        Device.lon,
        Device.altitude_m,
        START_DATE,
        END_DATE,
        model="ineichen",
        freq="1h",
        tz=TZ_LOCAL  # Fixed local timezone for all devices
    )
    df=ensure_weather_data(df)
    print(df.tz_convert(TZ_LOCAL).head())

    if not df.index.is_monotonic_increasing:
        print("⚠️  Timestamp order issue detected – sorting index")
        df = df.sort_index()
    
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
    plant_ac = compute_pv_ac(df, Device)
    
    # Compute timestep duration for energy integration
    dt_hours = (df.index[1] - df.index[0]).total_seconds() / 3600
    
    # Convert to fixed local timezone (Sri Lanka) for reporting/output only
    plant_ac_local = plant_ac.tz_convert(TZ_LOCAL)
    # Store a naive, local-time index for CSV output readability
    plant_ac_out = plant_ac_local.tz_localize(None)

    # Preview calculated AC power (local, first 5 rows)
    print("\nAC power preview (local time):")
    print(plant_ac_out.head())
    
    # =====================================================================
    # AGGREGATE TO DAILY ENERGY
    # =====================================================================
    
    print("\nAggregating to daily energy...")
    daily_power = aggregate_to_daily(plant_ac_local, dt_hours)
    
    # Save daily power data
    daily_output_path = Path("output") / f"{device_id}_daily_generation.csv"
    daily_power_out = daily_power.copy()
    daily_power_out.index = daily_power_out.index.tz_localize(None)  # Remove timezone for CSV
    daily_power_out.to_csv(daily_output_path)
    
    print(f"✓ Daily data saved: {daily_output_path}")
    print(f"  Total days: {len(daily_power)}")
    print(f"  Total energy: {daily_power['energy_kwh'].sum():.1f} kWh")
    print(f"  Average daily: {daily_power['energy_kwh'].mean():.1f} kWh")
    print("\nDaily power preview (first 5 days):")
    print(daily_power_out.head())
    
    # =====================================================================
    # INTEGRATE POWER TO ENERGY
    # =====================================================================
    
    # Integrate power to energy (kWh)
    energy = (plant_ac_local * dt_hours).sum()
    
    # =====================================================================
    # SAVE HOURLY OUTPUT
    # =====================================================================
    
    # Ensure output directory exists
    output_path = Path("output") / f"{device_id}_generation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    out = pd.DataFrame({"AC_kW": plant_ac_out})
    out.to_csv(output_path)
    
    # =====================================================================
    # SEND TELEMETRY DATA TO THINGSBOARD
    # =====================================================================
    
    print("\nSending telemetry data to ThingsBoard...")

    for i in range(3):
        print("Sending telemetry data on ",daily_power.index[i].strftime('%Y-%m-%d'))
        telemetry_data = {
            "ts": int(daily_power.index[i].timestamp() * 1000),
            "values": {
                "daily_energy_kwh_forcast": float(daily_power.iloc[i, 0])
            }
        }
        print(telemetry_data)
        write_device_telemetry(device_id, telemetry_data)
        print(f"✓ Sent telemetry: {telemetry_data}")
    print(f"✓ Sent telemetry: {telemetry_data}")

if __name__ == "__main__":
    level3_assets = get_assets_at_level(ROOT_ASSET_ID, TARGET_LEVEL)
    print(f"Total Level-{TARGET_LEVEL} assets found: {len(level3_assets)}")
    device_details = []
    for asset_id in level3_assets:
        devicesids = get_devices_from_asset(asset_id)
        devicenames= [get_device_name(did) for did in devicesids]
        for did, dname in zip(devicesids, devicenames):
            device_details.append([did, dname])
        print(f"Asset ID: {asset_id} | Devices found: {len(devicesids)}")
    print(len(device_details))
    print(device_details)

    for device in device_details:
        Solor_Power_Calculation(device)

