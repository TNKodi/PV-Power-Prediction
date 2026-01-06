#!/usr/bin/env python3
"""
================================================================================
MODEL EVALUATION SCRIPT
================================================================================

Evaluates all three PV prediction models on TEST DATA ONLY.
No data leakage: Test data is completely separate from training data.

Models evaluated:
1. Python Physics (ground truth - from test_data.csv)
2. JS Physics (run via Node.js on test weather data)
3. JS Surrogate (run via Node.js on test weather data, using trained coefficients)

USAGE:
------
  python evaluate_models.py --test-data ../../data/test_data.csv
  python evaluate_models.py --test-data ../../data/test_data.csv --skip-js

================================================================================
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================================
# PATHS
# =====================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
CONFIG_DIR = PROJECT_DIR / "config"


# =====================================================================
# LOAD DATA
# =====================================================================

def load_test_data(test_path):
    """
    Load test data from prepare_training_data.py.
    
    Expected columns:
    - period_end: UTC timestamp (index)
    - ghi, dni, dhi: Irradiance in W/m²
    - air_temp: Temperature in °C
    - wind_speed: Wind speed in m/s
    - ac_power_kw: Physics model prediction (ground truth)
    """
    print(f"\nLoading test data from: {test_path}")
    
    df = pd.read_csv(test_path)
    
    # Handle index
    if "period_end" in df.columns:
        df["period_end"] = pd.to_datetime(df["period_end"], utc=True)
        df = df.set_index("period_end")
    elif df.columns[0] == "Unnamed: 0" or "Unnamed" in df.columns[0]:
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "period_end"
    
    # Rename columns if needed
    if "wind_speed_10m" in df.columns:
        df = df.rename(columns={"wind_speed_10m": "wind_speed"})
    
    # Verify columns
    required_cols = ["ghi", "dni", "dhi", "air_temp", "wind_speed", "ac_power_kw"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"  Loaded {len(df)} test samples")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # Count unique days
    n_days = len(df.index.normalize().unique())
    print(f"  Test days: {n_days}")
    
    return df


def load_surrogate_coefficients():
    """Load trained surrogate coefficients."""
    coef_path = OUTPUT_DIR / "surrogate_coefficients.json"
    
    if not coef_path.exists():
        print(f"\n⚠️  Surrogate coefficients not found: {coef_path}")
        print("   Run fit_surrogate.py first.")
        return None
    
    with open(coef_path) as f:
        data = json.load(f)
    
    return data["coefficients"]


# =====================================================================
# PYTHON SURROGATE MODEL (for comparison with JS)
# =====================================================================

def run_python_surrogate(df, coefficients, inv_rating=55.0):
    """
    Run surrogate model in Python (same as JS implementation).
    Used to verify JS surrogate correctness.
    """
    c = coefficients
    
    # Feature engineering
    delta_T = df["air_temp"] - 25
    ghi = df["ghi"]
    dni = df["dni"]
    wind = df["wind_speed"]
    
    # Regression equation
    pred = (
        c["a0"] +
        c["a1"] * ghi +
        c["a2"] * ghi**2 +
        c["a3"] * dni +
        c["a4"] * ghi * delta_T +
        c["a5"] * ghi * wind
    )
    
    # Clip to physical limits
    pred = pred.clip(0, inv_rating)
    
    return pred


# =====================================================================
# JAVASCRIPT MODEL EVALUATION
# =====================================================================

def run_js_models(df_test):
    """
    Run JavaScript models on test data via Node.js.
    
    Creates a temporary CSV with test data, runs benchmark_js.js,
    and parses the results.
    """
    print("\n[2/4] Running JavaScript models via Node.js...")
    
    # Create temp CSV with test data in Solcast-compatible format
    temp_csv = DATA_DIR / "test_weather_temp.csv"
    
    # Prepare data for JS
    js_df = df_test[["ghi", "dni", "dhi", "air_temp", "wind_speed"]].copy()
    js_df["period_end"] = js_df.index.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    js_df["period"] = "PT60M"
    
    # Rename for Solcast format
    js_df = js_df.rename(columns={"wind_speed": "wind_speed_10m"})
    
    # Reorder columns
    js_df = js_df[["air_temp", "dhi", "dni", "ghi", "wind_speed_10m", "period_end", "period"]]
    
    js_df.to_csv(temp_csv, index=False)
    print(f"  Created temp CSV: {temp_csv}")
    
    # Create a modified benchmark script that uses our temp CSV
    benchmark_script = f'''
const fs = require('fs');
const path = require('path');

// Load surrogate model
const surrogateModel = require('../js_surrogate/surrogate_model.js');

// Load trained coefficients and override defaults
try {{
    const coefPath = '{(OUTPUT_DIR / "surrogate_coefficients.json").as_posix()}';
    const coefData = JSON.parse(fs.readFileSync(coefPath, 'utf8'));
    const trainedCoef = coefData.coefficients;
    
    // Override coefficients in surrogate model
    surrogateModel.REGRESSION_COEFFICIENTS.a0 = trainedCoef.a0;
    surrogateModel.REGRESSION_COEFFICIENTS.a1 = trainedCoef.a1;
    surrogateModel.REGRESSION_COEFFICIENTS.a2 = trainedCoef.a2;
    surrogateModel.REGRESSION_COEFFICIENTS.a3 = trainedCoef.a3;
    surrogateModel.REGRESSION_COEFFICIENTS.a4 = trainedCoef.a4;
    surrogateModel.REGRESSION_COEFFICIENTS.a5 = trainedCoef.a5;
    console.error('Loaded trained coefficients from:', coefPath);
}} catch (e) {{
    console.error('Warning: Could not load trained coefficients, using defaults');
}}

// Parse CSV
function parseCSV(filepath) {{
    const content = fs.readFileSync(filepath, 'utf8');
    const lines = content.trim().split('\\n');
    const headers = lines[0].split(',');
    const data = [];
    for (let i = 1; i < lines.length; i++) {{
        const values = lines[i].split(',');
        const row = {{}};
        headers.forEach((h, idx) => {{
            row[h.trim()] = values[idx] ? values[idx].trim() : '';
        }});
        data.push(row);
    }}
    return data;
}}

// Physics model config
const PHYSICS_CONFIG = {{
    lat: 8.342368984714714,
    lon: 80.37623529556957,
    tz_offset_h: 5.5,
    orientations: [
        {{tilt: 18, az: 148, mods: 18}},
        {{tilt: 18, az: -32, mods: 18}},
        {{tilt: 19, az: 55, mods: 36}},
        {{tilt: 19, az: -125, mods: 36}},
        {{tilt: 18, az: -125, mods: 36}},
        {{tilt: 18, az: 55, mods: 36}},
        {{tilt: 27, az: -125, mods: 18}},
        {{tilt: 27, az: 55, mods: 18}}
    ],
    mod_area: 2.556,
    mod_eff: 0.2153,
    temp_coeff: -0.00340,
    inv_rating_kw: 55,
    inv_eff: 0.98,
    inv_threshold_kw: 0.0,
    dc_losses: 0.9317,
    total_area: 552.096
}};

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

function clip(v, min, max) {{
    return v < min ? min : (v > max ? max : v);
}}

function solarPos(ts, lat, lon) {{
    const d = new Date(ts);
    const jd = d.getTime() / 86400000 + 2440587.5;
    const jc = (jd - 2451545) / 36525;
    
    const m = (357.52911 + jc * 35999.05029) * DEG2RAD;
    const c = (1.914602 - jc * 0.004817) * Math.sin(m) + 
            0.019993 * Math.sin(2 * m);
    const sunLon = (280.46646 + jc * 36000.76983 + c) * DEG2RAD;
    
    const obl = (23.439291 - jc * 0.0130042) * DEG2RAD;
    const dec = Math.asin(Math.sin(obl) * Math.sin(sunLon));
    
    const eot = 4 * (sunLon * RAD2DEG - Math.atan2(
        Math.cos(obl) * Math.sin(sunLon),
        Math.cos(sunLon)
    ) * RAD2DEG);
    
    const utcH = d.getUTCHours() + d.getUTCMinutes() / 60;
    const solarT = utcH + lon / 15 + eot / 60;
    const ha = (solarT - 12) * 15 * DEG2RAD;
    
    const latR = lat * DEG2RAD;
    const sinElev = Math.sin(latR) * Math.sin(dec) + 
                    Math.cos(latR) * Math.cos(dec) * Math.cos(ha);
    const elev = Math.asin(sinElev) * RAD2DEG;
    const zen = 90 - elev;
    
    const cosAz = (Math.sin(dec) - Math.sin(latR) * sinElev) / 
                  (Math.cos(latR) * Math.cos(Math.asin(sinElev)));
    let az = Math.acos(clip(cosAz, -1, 1)) * RAD2DEG;
    if (ha > 0) az = 360 - az;
    
    return {{ elev, zen, az }};
}}

function calcPhysics(ts, ghi, dni, dhi, tAir, wind) {{
    const sun = solarPos(ts, PHYSICS_CONFIG.lat, PHYSICS_CONFIG.lon);
    if (sun.elev <= 0 || ghi <= 0) return {{ ac: 0, dc: 0 }};
    
    const cosZ = Math.cos(sun.zen * DEG2RAD);
    let totalDC = 0;
    
    for (const o of PHYSICS_CONFIG.orientations) {{
        const tiltR = o.tilt * DEG2RAD;
        const azR = o.az * DEG2RAD;
        const sunAzR = sun.az * DEG2RAD;
        
        const cosAOI = Math.sin(sun.elev * DEG2RAD) * Math.cos(tiltR) +
                       Math.cos(sun.elev * DEG2RAD) * Math.sin(tiltR) *
                       Math.cos(sunAzR - azR);
        const aoi = Math.acos(clip(cosAOI, -1, 1)) * RAD2DEG;
        
        const poa_beam = (aoi < 90 && dni > 0) ? dni * cosAOI : 0;
        const poa_diff = dhi * (1 + Math.cos(tiltR)) / 2;
        const poa_gnd = ghi * 0.2 * (1 - Math.cos(tiltR)) / 2;
        let poa = poa_beam + poa_diff + poa_gnd;
        if (poa < 0) poa = 0;
        
        const tCell = tAir + poa * 0.03;
        const tempFac = 1 + PHYSICS_CONFIG.temp_coeff * (tCell - 25);
        
        const areaFrac = o.mods / 216;
        const dcKwM2 = poa * PHYSICS_CONFIG.mod_eff * tempFac * PHYSICS_CONFIG.dc_losses / 1000;
        
        totalDC += dcKwM2 * PHYSICS_CONFIG.total_area * areaFrac;
    }}
    
    const ac = Math.min(totalDC * PHYSICS_CONFIG.inv_eff, PHYSICS_CONFIG.inv_rating_kw);
    return {{ ac, dc: totalDC }};
}}

// Main
const rawData = parseCSV('{temp_csv.as_posix()}');
const results = {{
    physics_js: [],
    surrogate_js: [],
    timing: {{
        physics_js: {{ total_ms: 0, predictions: 0 }},
        surrogate_js: {{ total_ms: 0, predictions: 0 }}
    }}
}};

// Run physics model
let t0 = Date.now();
for (const row of rawData) {{
    const result = calcPhysics(
        row.period_end,
        parseFloat(row.ghi) || 0,
        parseFloat(row.dni) || 0,
        parseFloat(row.dhi) || 0,
        parseFloat(row.air_temp) || 25,
        parseFloat(row.wind_speed_10m) || 1
    );
    results.physics_js.push({{
        timestamp: row.period_end,
        ac_power_kw: result.ac
    }});
}}
results.timing.physics_js.total_ms = Date.now() - t0;
results.timing.physics_js.predictions = rawData.length;

// Run surrogate model
t0 = Date.now();
for (const row of rawData) {{
    const inputs = {{
        ghi: parseFloat(row.ghi) || 0,
        dni: parseFloat(row.dni) || 0,
        dhi: parseFloat(row.dhi) || 0,
        airTemp: parseFloat(row.air_temp) || 25,
        windSpeed: parseFloat(row.wind_speed_10m) || 1
    }};
    const result = surrogateModel.predictPV(inputs);
    results.surrogate_js.push({{
        timestamp: row.period_end,
        ac_power_kw: result.powerKW || 0
    }});
}}
results.timing.surrogate_js.total_ms = Date.now() - t0;
results.timing.surrogate_js.predictions = rawData.length;

// Output results to file
const outputPath = '{(OUTPUT_DIR / "test_js_results.json").as_posix()}';
fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));

// Output to stdout for Python to parse (single line JSON)
console.log('JS_RESULT_START');
console.log(JSON.stringify(results));
console.log('JS_RESULT_END');
'''
    
    # Write temp script
    temp_script = SCRIPT_DIR / "temp_eval_js.js"
    with open(temp_script, "w") as f:
        f.write(benchmark_script)
    
    try:
        result = subprocess.run(
            ["node", str(temp_script)],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return None, None
        
        # Parse results - find JSON between markers
        stdout = result.stdout
        if "JS_RESULT_START" in stdout:
            json_start = stdout.find("JS_RESULT_START") + len("JS_RESULT_START")
            json_end = stdout.find("JS_RESULT_END")
            json_str = stdout[json_start:json_end].strip()
        else:
            json_str = stdout.strip()
        
        js_results = json.loads(json_str)
        
        # Convert to DataFrames
        physics_js = pd.DataFrame(js_results["physics_js"])
        physics_js["timestamp"] = pd.to_datetime(physics_js["timestamp"], utc=True)
        physics_js = physics_js.set_index("timestamp")["ac_power_kw"]
        
        surrogate_js = pd.DataFrame(js_results["surrogate_js"])
        surrogate_js["timestamp"] = pd.to_datetime(surrogate_js["timestamp"], utc=True)
        surrogate_js = surrogate_js.set_index("timestamp")["ac_power_kw"]
        
        print(f"  JS Physics: {len(physics_js)} predictions in {js_results['timing']['physics_js']['total_ms']}ms")
        print(f"  JS Surrogate: {len(surrogate_js)} predictions in {js_results['timing']['surrogate_js']['total_ms']}ms")
        
        return physics_js, surrogate_js
        
    except Exception as e:
        print(f"  ERROR running JS: {e}")
        return None, None
    
    finally:
        # Cleanup
        if temp_script.exists():
            temp_script.unlink()
        if temp_csv.exists():
            temp_csv.unlink()


# =====================================================================
# METRICS CALCULATION
# =====================================================================

def compute_metrics(actual, predicted, name):
    """Compute comprehensive evaluation metrics."""
    
    # Align indices
    common = actual.index.intersection(predicted.index)
    y_true = actual.loc[common].values
    y_pred = predicted.loc[common].values
    
    if len(common) == 0:
        return {"name": name, "error": "No overlapping data"}
    
    # Handle NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    
    # MAPE (excluding near-zero values)
    daytime_mask = y_true > 1.0
    if daytime_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[daytime_mask] - y_pred[daytime_mask]) / y_true[daytime_mask])) * 100
    else:
        mape = np.nan
    
    # Energy metrics
    total_actual = y_true.sum()
    total_predicted = y_pred.sum()
    energy_error = total_predicted - total_actual
    energy_error_pct = (energy_error / total_actual * 100) if total_actual > 0 else 0
    
    # Bias
    bias = np.mean(y_pred - y_true)
    
    metrics = {
        "name": name,
        "n_samples": len(y_true),
        "mae_kw": float(mae),
        "rmse_kw": float(rmse),
        "r2": float(r2),
        "mape_pct": float(mape) if not np.isnan(mape) else None,
        "bias_kw": float(bias),
        "total_actual_kwh": float(total_actual),
        "total_predicted_kwh": float(total_predicted),
        "energy_error_kwh": float(energy_error),
        "energy_error_pct": float(energy_error_pct)
    }
    
    return metrics


def print_metrics_table(metrics_list):
    """Print metrics in a nice table format."""
    
    print("\n" + "=" * 90)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 90)
    print(f"\n{'Model':<20} {'MAE (kW)':<12} {'RMSE (kW)':<12} {'R²':<10} {'MAPE (%)':<12} {'Energy Err (%)':<15}")
    print("-" * 90)
    
    for m in metrics_list:
        if "error" in m:
            print(f"{m['name']:<20} ERROR: {m['error']}")
        else:
            mape_str = f"{m['mape_pct']:.2f}" if m['mape_pct'] is not None else "N/A"
            print(f"{m['name']:<20} {m['mae_kw']:<12.2f} {m['rmse_kw']:<12.2f} {m['r2']:<10.4f} {mape_str:<12} {m['energy_error_pct']:<15.2f}")
    
    print("=" * 90)


# =====================================================================
# VISUALIZATION
# =====================================================================

def create_comparison_plots(df_test, predictions_dict, output_dir):
    """Create comparison plots for all models."""
    
    print("\n[4/4] Generating comparison plots...")
    
    actual = df_test["ac_power_kw"]
    
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))
    
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    colors = {"Python Surrogate": "blue", "JS Physics": "orange", "JS Surrogate": "green"}
    
    for i, (name, pred) in enumerate(predictions_dict.items()):
        color = colors.get(name, "gray")
        
        # Scatter plot
        ax = axes[0, i]
        common = actual.index.intersection(pred.index)
        y_true = actual.loc[common]
        y_pred = pred.loc[common]
        
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, c=color)
        ax.plot([0, 55], [0, 55], 'r--', linewidth=2, label='Perfect Fit')
        ax.set_xlabel('Physics Model (kW)', fontsize=10)
        ax.set_ylabel(f'{name} (kW)', fontsize=10)
        ax.set_title(f'{name}\nvs Physics (Ground Truth)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 60)
        
        # Residual histogram
        ax = axes[1, i]
        residuals = y_pred - y_true
        ax.hist(residuals, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual (kW)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        mae = np.mean(np.abs(residuals))
        ax.set_title(f'Residuals (MAE = {mae:.2f} kW)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "test_evaluation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    
    plt.close()


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all models on test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_models.py --test-data ../../data/test_data.csv
  python evaluate_models.py --test-data ../../data/test_data.csv --skip-js
        """
    )
    
    parser.add_argument(
        "--test-data",
        default=None,
        help="Path to test data CSV from prepare_training_data.py"
    )
    parser.add_argument(
        "--skip-js",
        action="store_true",
        help="Skip JavaScript model evaluation"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ../../output)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MODEL EVALUATION ON TEST DATA")
    print("=" * 70)
    print("\n⚠️  This evaluates models on TEST data only (no data leakage)")
    
    # Determine paths
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    
    # Load test data
    if args.test_data:
        test_path = Path(args.test_data)
    else:
        test_path = DATA_DIR / "test_data.csv"
        if not test_path.exists():
            print(f"\n⚠️  Test data not found: {test_path}")
            print("   Run prepare_training_data.py first.")
            sys.exit(1)
    
    df_test = load_test_data(test_path)
    
    # Python Physics is already in test data as "ac_power_kw"
    actual = df_test["ac_power_kw"]
    
    # =====================================================================
    # EVALUATE MODELS
    # =====================================================================
    
    print("\n[1/4] Loading trained coefficients...")
    coefficients = load_surrogate_coefficients()
    
    predictions = {}
    metrics_list = []
    
    # Surrogate model verification (Python reference - same as JS Surrogate)
    if coefficients:
        print("\n  Running Surrogate model (Python reference)...")
        pred_py_surr = run_python_surrogate(df_test, coefficients)
        predictions["Surrogate (verify)"] = pred_py_surr
        # Don't add to metrics - JS Surrogate is the real one
    
    # JavaScript Models
    if not args.skip_js:
        physics_js, surrogate_js = run_js_models(df_test)
        
        if physics_js is not None:
            predictions["JS Physics"] = physics_js
            m = compute_metrics(actual, physics_js, "JS Physics")
            metrics_list.append(m)
        
        if surrogate_js is not None:
            predictions["JS Surrogate"] = surrogate_js
            m = compute_metrics(actual, surrogate_js, "JS Surrogate")
            metrics_list.append(m)
    else:
        print("\n[2/4] Skipping JavaScript models (--skip-js)")
    
    # =====================================================================
    # RESULTS
    # =====================================================================
    
    print("\n[3/4] Computing metrics...")
    print_metrics_table(metrics_list)
    
    # Save metrics to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "test_evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "test_samples": len(df_test),
            "test_days": len(df_test.index.normalize().unique()),
            "models": metrics_list
        }, f, indent=2)
    print(f"\nSaved: {metrics_path}")
    
    # Create plots
    if predictions:
        create_comparison_plots(df_test, predictions, output_dir)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nKey findings:")
    
    for m in metrics_list:
        if "error" not in m:
            status = "✓" if m["mae_kw"] < 5 else "⚠️"
            print(f"  {status} {m['name']}: MAE = {m['mae_kw']:.2f} kW, Energy Error = {m['energy_error_pct']:.2f}%")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

