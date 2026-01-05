"""
Model Benchmark Orchestrator

Compares three PV prediction models:
1. Python Physics (pvlib + Perez + SAPM)
2. JS Physics (simplified Perez)
3. JS Surrogate (regression-based)

Against actual Meteocontrol generation data.

Usage: python run_benchmark.py
"""

import pandas as pd
import numpy as np
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Timezone utilities for consistent handling
from timezone_utils import load_meteocontrol, to_local, to_utc, LOCAL_TZ

# =====================================================================
# CONFIGURATION
# =====================================================================

TIMEZONE = LOCAL_TZ  # Asia/Colombo (UTC+5:30)
VALID_DAYS = [11, 12, 13, 14, 15]  # December 2025 complete days

# Paths
# Path: scripts/shared/ -> ../../data, ../../output
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent  # Power Prediction/
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

# =====================================================================
# PYTHON PHYSICS MODEL (inline for benchmark)
# =====================================================================

import pvlib
from pvlib.location import Location

# Plant configuration (from config/plant_config.json)
PLANT_CONFIG = {
    "lat": 8.342368984714714,
    "lon": 80.37623529556957,
    "altitude": 88,
    "orientations": [
        {"tilt": 18, "azimuth": 148, "module_count": 18},
        {"tilt": 18, "azimuth": -32, "module_count": 18},
        {"tilt": 19, "azimuth": 55, "module_count": 36},
        {"tilt": 19, "azimuth": -125, "module_count": 36},
        {"tilt": 18, "azimuth": -125, "module_count": 36},
        {"tilt": 18, "azimuth": 55, "module_count": 36},
        {"tilt": 27, "azimuth": -125, "module_count": 18},
        {"tilt": 27, "azimuth": 55, "module_count": 18}
    ],
    "module_area": 2.556,
    "module_efficiency_stc": 0.2153,
    "gamma_p": -0.00340,
    "inv_ac_rating_kw": 55.0,
    "albedo": 0.20,
    "dc_loss_factor": 0.9317,
    "ac_wiring_loss": 0.003,
    "iam_angles": [0, 25, 45, 60, 65, 70, 75, 80, 90],
    "iam_values": [1.000, 1.000, 0.995, 0.962, 0.936, 0.903, 0.851, 0.754, 0.000]
}


def run_python_physics(df_weather):
    """Run Python physics model on weather data."""
    
    lat = PLANT_CONFIG["lat"]
    lon = PLANT_CONFIG["lon"]
    altitude = PLANT_CONFIG["altitude"]
    orientations = PLANT_CONFIG["orientations"]
    module_area = PLANT_CONFIG["module_area"]
    module_efficiency_stc = PLANT_CONFIG["module_efficiency_stc"]
    gamma_p = PLANT_CONFIG["gamma_p"]
    inv_ac_rating_kw = PLANT_CONFIG["inv_ac_rating_kw"]
    albedo = PLANT_CONFIG["albedo"]
    dc_loss_factor = PLANT_CONFIG["dc_loss_factor"]
    ac_wiring_loss = PLANT_CONFIG["ac_wiring_loss"]
    iam_angles = np.array(PLANT_CONFIG["iam_angles"])
    iam_values = np.array(PLANT_CONFIG["iam_values"])
    
    total_module_area = sum(o["module_count"] * module_area for o in orientations)
    
    # SAPM thermal parameters
    sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
    
    site = Location(lat, lon, altitude=altitude)
    solpos = site.get_solarposition(df_weather.index)
    dni_extra = pvlib.irradiance.get_extra_radiation(df_weather.index)
    
    plant_ac = pd.Series(0.0, index=df_weather.index)
    
    for o in orientations:
        tilt = o["tilt"]
        azimuth = o["azimuth"]
        area_fraction = (o["module_count"] * module_area) / total_module_area
        
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
        
        aoi = pvlib.irradiance.aoi(tilt, azimuth, solpos["zenith"], solpos["azimuth"])
        iam = np.interp(aoi, iam_angles, iam_values)
        poa_optical = poa * iam
        
        cell_temp = pvlib.temperature.sapm_cell(
            poa, df_weather["air_temp"], df_weather["wind_speed"], **sapm_params
        )
        
        pdc_kwm2 = poa_optical * module_efficiency_stc / 1000
        pdc_kwm2 *= (1 + gamma_p * (cell_temp - 25))
        pdc_kwm2 *= dc_loss_factor
        
        area_i = total_module_area * area_fraction
        pdc_total_kw = pdc_kwm2 * area_i
        
        # Inverter efficiency (simplified)
        pac_kw = pdc_total_kw * 0.98
        plant_ac += pac_kw
    
    plant_ac = plant_ac.clip(upper=inv_ac_rating_kw)
    plant_ac = plant_ac * (1 - ac_wiring_loss)
    
    return plant_ac


# =====================================================================
# DATA LOADING
# =====================================================================

def load_solcast_data():
    """Load and prepare Solcast irradiance data."""
    df = pd.read_csv(DATA_DIR / "solcast_irradiance.csv")
    
    df["period_end"] = pd.to_datetime(df["period_end"], utc=True)
    df = df.set_index("period_end").sort_index()
    
    # Rename columns
    if "wind_speed_10m" in df.columns:
        df["wind_speed"] = df["wind_speed_10m"]
    
    return df


def load_meteocontrol_actual():
    """
    Load actual Meteocontrol daily energy data.
    
    Uses timezone_utils to properly handle timezone:
    - Raw data is in Asia/Colombo (GMT+5:30)
    - Returns DataFrame with timezone-aware index
    """
    df = load_meteocontrol(DATA_DIR / "meteocontrol_actual.csv")
    return df[["day", "actual_kwh"]]


# =====================================================================
# METRICS CALCULATION
# =====================================================================

def calculate_metrics(predicted, actual, name):
    """Calculate accuracy metrics."""
    
    # Align on common index
    common = predicted.index.intersection(actual.index)
    pred = predicted.loc[common]
    act = actual.loc[common]
    
    if len(common) == 0:
        return {"name": name, "error": "No overlapping data"}
    
    errors = pred - act
    abs_errors = np.abs(errors)
    pct_errors = np.where(act > 0, (errors / act) * 100, 0)
    
    metrics = {
        "name": name,
        "n_days": len(common),
        "mae_kwh": float(abs_errors.mean()),
        "rmse_kwh": float(np.sqrt((errors ** 2).mean())),
        "mape_pct": float(np.abs(pct_errors).mean()),
        "bias_pct": float(pct_errors.mean()),
        "total_predicted_kwh": float(pred.sum()),
        "total_actual_kwh": float(act.sum()),
        "total_error_pct": float((pred.sum() - act.sum()) / act.sum() * 100)
    }
    
    return metrics


# =====================================================================
# MAIN BENCHMARK
# =====================================================================

def run_benchmark():
    print("=" * 70)
    print("MODEL BENCHMARK COMPARISON")
    print("=" * 70)
    
    # =====================================================================
    # LOAD DATA
    # =====================================================================
    print("\n[1/5] Loading data...")
    
    df_weather = load_solcast_data()
    df_actual = load_meteocontrol_actual()
    
    print(f"  Solcast data: {len(df_weather)} hourly records")
    print(f"  Date range: {df_weather.index.min()} to {df_weather.index.max()}")
    print(f"  Meteocontrol data: {len(df_actual)} daily records")
    
    # =====================================================================
    # RUN PYTHON PHYSICS MODEL
    # =====================================================================
    print("\n[2/5] Running Python Physics Model...")
    
    start_time = time.perf_counter()
    python_hourly = run_python_physics(df_weather)
    python_time = time.perf_counter() - start_time
    
    # Convert to local timezone and aggregate to daily
    # Keep timezone-aware for proper comparison with actual data
    python_local = python_hourly.tz_convert(TIMEZONE)
    python_daily = python_local.resample("D").sum()
    
    # Filter to valid complete days only
    python_daily = python_daily[python_daily.index.day.isin(VALID_DAYS)]
    
    print(f"  Total time: {python_time*1000:.2f} ms")
    print(f"  Avg per prediction: {python_time/len(df_weather)*1000:.3f} ms")
    print(f"  Valid days: {list(python_daily.index.day)}")
    
    # =====================================================================
    # RUN JAVASCRIPT MODELS
    # =====================================================================
    print("\n[3/5] Running JavaScript Models (via Node.js)...")
    
    try:
        result = subprocess.run(
            ["node", "benchmark_js.js"],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            js_results = None
        else:
            print(result.stdout)
            
            # Load JS results
            js_results_path = OUTPUT_DIR / "js_benchmark_results.json"
            with open(js_results_path) as f:
                js_results = json.load(f)
                
    except FileNotFoundError:
        print("  ERROR: Node.js not found. Install Node.js to run JS benchmarks.")
        js_results = None
    except subprocess.TimeoutExpired:
        print("  ERROR: JS benchmark timed out")
        js_results = None
    
    # Process JS results if available
    js_physics_daily = None
    js_surrogate_daily = None
    js_timing = {}
    
    if js_results:
        # Convert JS physics results to daily
        # Keep timezone-aware for proper comparison
        js_physics_df = pd.DataFrame(js_results["physics_js"])
        js_physics_df["timestamp"] = pd.to_datetime(js_physics_df["timestamp"], utc=True)
        js_physics_df = js_physics_df.set_index("timestamp")
        js_physics_local = js_physics_df["ac_power_kw"].tz_convert(TIMEZONE)
        js_physics_daily = js_physics_local.resample("D").sum()
        js_physics_daily = js_physics_daily[js_physics_daily.index.day.isin(VALID_DAYS)]
        
        # Convert JS surrogate results to daily
        js_surrogate_df = pd.DataFrame(js_results["surrogate_js"])
        js_surrogate_df["timestamp"] = pd.to_datetime(js_surrogate_df["timestamp"], utc=True)
        js_surrogate_df = js_surrogate_df.set_index("timestamp")
        js_surrogate_local = js_surrogate_df["ac_power_kw"].tz_convert(TIMEZONE)
        js_surrogate_daily = js_surrogate_local.resample("D").sum()
        js_surrogate_daily = js_surrogate_daily[js_surrogate_daily.index.day.isin(VALID_DAYS)]
        
        js_timing = js_results["timing"]
    
    # =====================================================================
    # CALCULATE METRICS
    # =====================================================================
    print("\n[4/5] Calculating metrics...")
    
    # Filter actual data to valid days
    actual_valid = df_actual[df_actual["day"].isin(VALID_DAYS)]["actual_kwh"]
    
    metrics = []
    
    # Python Physics
    python_metrics = calculate_metrics(python_daily, actual_valid, "Python Physics")
    python_metrics["avg_time_ms"] = python_time / len(df_weather) * 1000
    python_metrics["total_time_ms"] = python_time * 1000
    metrics.append(python_metrics)
    
    # JS Physics
    if js_physics_daily is not None:
        js_phys_metrics = calculate_metrics(js_physics_daily, actual_valid, "JS Physics")
        js_phys_metrics["avg_time_ms"] = js_timing["physics_js"]["total_ms"] / js_timing["physics_js"]["predictions"]
        js_phys_metrics["total_time_ms"] = js_timing["physics_js"]["total_ms"]
        metrics.append(js_phys_metrics)
    
    # JS Surrogate
    if js_surrogate_daily is not None:
        js_surr_metrics = calculate_metrics(js_surrogate_daily, actual_valid, "JS Surrogate")
        js_surr_metrics["avg_time_ms"] = js_timing["surrogate_js"]["total_ms"] / js_timing["surrogate_js"]["predictions"]
        js_surr_metrics["total_time_ms"] = js_timing["surrogate_js"]["total_ms"]
        metrics.append(js_surr_metrics)
    
    # =====================================================================
    # GENERATE OUTPUTS
    # =====================================================================
    print("\n[5/5] Generating outputs...")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save metrics JSON
    metrics_path = OUTPUT_DIR / "benchmark_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved: {metrics_path}")
    
    # Create comparison DataFrame for plotting
    comparison = pd.DataFrame({
        "Actual": actual_valid,
        "Python Physics": python_daily
    })
    
    if js_physics_daily is not None:
        comparison["JS Physics"] = js_physics_daily
    if js_surrogate_daily is not None:
        comparison["JS Surrogate"] = js_surrogate_daily
    
    comparison = comparison.dropna()
    
    # Save comparison CSV
    comparison_path = OUTPUT_DIR / "benchmark_results.csv"
    comparison.to_csv(comparison_path)
    print(f"  Saved: {comparison_path}")
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Daily Energy Comparison
    ax = axes[0, 0]
    x = range(len(comparison))
    width = 0.2
    
    ax.bar([i - 1.5*width for i in x], comparison["Actual"], width, label="Actual", color="green", alpha=0.7)
    ax.bar([i - 0.5*width for i in x], comparison["Python Physics"], width, label="Python Physics", color="blue", alpha=0.7)
    
    if "JS Physics" in comparison.columns:
        ax.bar([i + 0.5*width for i in x], comparison["JS Physics"], width, label="JS Physics", color="orange", alpha=0.7)
    if "JS Surrogate" in comparison.columns:
        ax.bar([i + 1.5*width for i in x], comparison["JS Surrogate"], width, label="JS Surrogate", color="red", alpha=0.7)
    
    ax.set_xlabel("Day of December 2025")
    ax.set_ylabel("Daily Energy (kWh)")
    ax.set_title("Daily Energy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Dec {d}" for d in comparison.index.day])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error by Model
    ax = axes[0, 1]
    model_names = [m["name"] for m in metrics]
    maes = [m.get("mae_kwh", 0) for m in metrics]
    colors = ["blue", "orange", "red"][:len(metrics)]
    
    ax.bar(model_names, maes, color=colors, alpha=0.7)
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("Mean Absolute Error by Model")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Speed Comparison
    ax = axes[1, 0]
    times = [m.get("avg_time_ms", 0) for m in metrics]
    
    ax.bar(model_names, times, color=colors, alpha=0.7)
    ax.set_ylabel("Avg Time per Prediction (ms)")
    ax.set_title("Prediction Speed Comparison")
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary Table
    ax = axes[1, 1]
    ax.axis("off")
    
    table_data = []
    headers = ["Metric", "Python Physics", "JS Physics", "JS Surrogate"]
    
    row_labels = ["MAE (kWh)", "MAPE (%)", "Bias (%)", "Total Error (%)", "Avg Time (ms)"]
    
    for row_label in row_labels:
        row = [row_label]
        for m in metrics:
            if row_label == "MAE (kWh)":
                row.append(f"{m.get('mae_kwh', 'N/A'):.1f}" if isinstance(m.get('mae_kwh'), (int, float)) else "N/A")
            elif row_label == "MAPE (%)":
                row.append(f"{m.get('mape_pct', 'N/A'):.1f}" if isinstance(m.get('mape_pct'), (int, float)) else "N/A")
            elif row_label == "Bias (%)":
                row.append(f"{m.get('bias_pct', 'N/A'):.1f}" if isinstance(m.get('bias_pct'), (int, float)) else "N/A")
            elif row_label == "Total Error (%)":
                row.append(f"{m.get('total_error_pct', 'N/A'):.1f}" if isinstance(m.get('total_error_pct'), (int, float)) else "N/A")
            elif row_label == "Avg Time (ms)":
                row.append(f"{m.get('avg_time_ms', 'N/A'):.3f}" if isinstance(m.get('avg_time_ms'), (int, float)) else "N/A")
        
        # Pad row if fewer models
        while len(row) < 4:
            row.append("N/A")
        
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers[:len(metrics)+1], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("Benchmark Summary", pad=20, fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "benchmark_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {plot_path}")
    
    # Generate markdown report
    report = generate_report(metrics, comparison)
    report_path = OUTPUT_DIR / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nComparison period: December 11-15, 2025 ({len(comparison)} complete days)")
    print(f"Timezone: {TIMEZONE}")
    print()
    
    print(f"{'Model':<20} {'MAE (kWh)':<12} {'MAPE (%)':<12} {'Bias (%)':<12} {'Avg Time (ms)':<15}")
    print("-" * 70)
    
    for m in metrics:
        name = m["name"]
        mae = f"{m.get('mae_kwh', 'N/A'):.1f}" if isinstance(m.get('mae_kwh'), (int, float)) else "N/A"
        mape = f"{m.get('mape_pct', 'N/A'):.1f}" if isinstance(m.get('mape_pct'), (int, float)) else "N/A"
        bias = f"{m.get('bias_pct', 'N/A'):.1f}" if isinstance(m.get('bias_pct'), (int, float)) else "N/A"
        time_ms = f"{m.get('avg_time_ms', 'N/A'):.3f}" if isinstance(m.get('avg_time_ms'), (int, float)) else "N/A"
        
        print(f"{name:<20} {mae:<12} {mape:<12} {bias:<12} {time_ms:<15}")
    
    print("=" * 70)


def generate_report(metrics, comparison):
    """Generate markdown benchmark report."""
    
    report = f"""# Model Benchmark Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

Comparison of three PV prediction models against actual Meteocontrol generation data.

**Comparison Period:** December 11-15, 2025 (5 complete days)
**Timezone:** Asia/Colombo (UTC+5:30)

## Results

| Model | MAE (kWh) | MAPE (%) | Bias (%) | Total Error (%) | Avg Time (ms) |
|-------|-----------|----------|----------|-----------------|---------------|
"""
    
    for m in metrics:
        name = m["name"]
        mae = f"{m.get('mae_kwh', 'N/A'):.1f}" if isinstance(m.get('mae_kwh'), (int, float)) else "N/A"
        mape = f"{m.get('mape_pct', 'N/A'):.1f}" if isinstance(m.get('mape_pct'), (int, float)) else "N/A"
        bias = f"{m.get('bias_pct', 'N/A'):.1f}" if isinstance(m.get('bias_pct'), (int, float)) else "N/A"
        total_err = f"{m.get('total_error_pct', 'N/A'):.1f}" if isinstance(m.get('total_error_pct'), (int, float)) else "N/A"
        time_ms = f"{m.get('avg_time_ms', 'N/A'):.3f}" if isinstance(m.get('avg_time_ms'), (int, float)) else "N/A"
        
        report += f"| {name} | {mae} | {mape} | {bias} | {total_err} | {time_ms} |\n"
    
    report += f"""
## Daily Energy Comparison (kWh)

| Date | Actual | Python Physics | JS Physics | JS Surrogate |
|------|--------|----------------|------------|--------------|
"""
    
    for idx, row in comparison.iterrows():
        date = idx.strftime("%Y-%m-%d")
        actual = f"{row['Actual']:.1f}"
        python = f"{row['Python Physics']:.1f}"
        js_phys = f"{row.get('JS Physics', 'N/A'):.1f}" if pd.notna(row.get('JS Physics')) else "N/A"
        js_surr = f"{row.get('JS Surrogate', 'N/A'):.1f}" if pd.notna(row.get('JS Surrogate')) else "N/A"
        
        report += f"| {date} | {actual} | {python} | {js_phys} | {js_surr} |\n"
    
    report += """
## Notes

- **Python Physics**: Full pvlib model with Perez POA transposition, SAPM thermal model
- **JS Physics**: Simplified Perez approximation, designed for ThingsBoard
- **JS Surrogate**: Regression-based model (coefficients may need training)
- All timestamps converted to Asia/Colombo before daily aggregation
- Only complete days (Dec 11-15) included in comparison
"""
    
    return report


if __name__ == "__main__":
    run_benchmark()

