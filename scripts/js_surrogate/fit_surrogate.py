#!/usr/bin/env python3
"""
================================================================================
PV SURROGATE MODEL - COEFFICIENT FITTING SCRIPT
================================================================================

Trains a linear regression surrogate model on training data ONLY.
No data leakage: Uses separate train/test split from prepare_training_data.py.

USAGE:
------
  # Train on pre-split training data (recommended)
  python fit_surrogate.py --train-data ../../data/train_data.csv

  # Legacy mode: Use old Solcast + physics model output (for backwards compat)
  python fit_surrogate.py --legacy

================================================================================
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

def load_training_data(train_path):
    """
    Load pre-split training data from prepare_training_data.py.
    
    Expected columns:
    - period_end (or index): UTC timestamp
    - ghi, dni, dhi: Irradiance in W/m²
    - air_temp: Temperature in °C
    - wind_speed: Wind speed in m/s
    - ac_power_kw: Physics model prediction (ground truth for training)
    """
    print(f"\nLoading training data from: {train_path}")
    
    df = pd.read_csv(train_path)
    
    # Handle index
    if "period_end" in df.columns:
        df["period_end"] = pd.to_datetime(df["period_end"], utc=True)
        df = df.set_index("period_end")
    elif df.columns[0] == "Unnamed: 0" or "Unnamed" in df.columns[0]:
        # First column is index
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "period_end"
    
    # Rename columns if needed
    if "wind_speed_10m" in df.columns:
        df = df.rename(columns={"wind_speed_10m": "wind_speed"})
    
    # Verify required columns
    required_cols = ["ghi", "dni", "dhi", "air_temp", "wind_speed", "ac_power_kw"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    
    print(f"  Loaded {len(df)} samples")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def load_legacy_data():
    """
    Load data using legacy method (Solcast + physics output).
    FOR BACKWARDS COMPATIBILITY ONLY.
    """
    print("\n⚠️  LEGACY MODE: Using old data loading method")
    print("   This may cause data leakage if train/test split is not managed externally.")
    
    input_csv = DATA_DIR / "solcast_irradiance.csv"
    output_csv = OUTPUT_DIR / "pv_generation.csv"
    
    print(f"\nLoading input data: {input_csv}")
    df_input = pd.read_csv(input_csv)
    
    print(f"Loading output data: {output_csv}")
    df_output = pd.read_csv(output_csv)
    
    # Timestamp normalization
    df_input['period_end'] = pd.to_datetime(df_input['period_end'], utc=True)
    df_output['period_end'] = pd.to_datetime(df_output.iloc[:, 0])  # First column is timestamp
    
    # Convert PV output (Asia/Colombo) → UTC
    if df_output['period_end'].dt.tz is not None:
        df_output['period_end'] = df_output['period_end'].dt.tz_convert('UTC')
    else:
        df_output['period_end'] = df_output['period_end'].dt.tz_localize('Asia/Colombo').dt.tz_convert('UTC')
    
    # Snap to hour end
    df_output['period_end'] = df_output['period_end'].dt.ceil('H')
    
    # Get AC_kW column
    ac_col = [c for c in df_output.columns if 'AC' in c or 'kW' in c or 'power' in c.lower()]
    if ac_col:
        df_output = df_output.rename(columns={ac_col[0]: 'ac_power_kw'})
    
    # Merge
    df = pd.merge(df_input, df_output[['period_end', 'ac_power_kw']], on='period_end', how='inner')
    
    # Rename columns
    if 'wind_speed_10m' in df.columns:
        df['wind_speed'] = df['wind_speed_10m']
    
    df = df.set_index('period_end')
    
    print(f"  Merged dataset: {len(df)} samples")
    
    return df


# =====================================================================
# FEATURE ENGINEERING
# =====================================================================

def engineer_features(df):
    """
    Create regression features from weather data.
    
    Features:
    - ghi: Linear irradiance term
    - ghi_squared: Quadratic term (captures nonlinearity at high irradiance)
    - dni: Direct normal contribution
    - ghi_temp: Temperature interaction
    - ghi_wind: Wind cooling interaction
    """
    print("\nEngineering features...")
    
    # Temperature deviation from STC (25°C)
    df = df.copy()
    df['delta_T'] = df['air_temp'] - 25
    
    # Feature engineering
    df['ghi_squared'] = df['ghi'] ** 2
    df['ghi_temp'] = df['ghi'] * df['delta_T']
    df['ghi_wind'] = df['ghi'] * df['wind_speed']
    
    # Feature matrix
    feature_cols = ['ghi', 'ghi_squared', 'dni', 'ghi_temp', 'ghi_wind']
    X = df[feature_cols].values
    y = df['ac_power_kw'].values
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Target vector shape: {y.shape}")
    
    return X, y, df


# =====================================================================
# MODEL FITTING
# =====================================================================

def fit_model(X, y):
    """Fit linear regression model."""
    print("\nFitting regression model...")
    
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    
    # Extract coefficients
    coefficients = {
        'a0': float(model.intercept_),
        'a1': float(model.coef_[0]),  # GHI
        'a2': float(model.coef_[1]),  # GHI²
        'a3': float(model.coef_[2]),  # DNI
        'a4': float(model.coef_[3]),  # GHI × ΔT
        'a5': float(model.coef_[4]),  # GHI × wind
    }
    
    print("✓ Regression complete")
    print(f"\n{'='*60}")
    print("FITTED COEFFICIENTS")
    print(f"{'='*60}")
    print(f"  a0 (intercept):        {coefficients['a0']:12.6f}")
    print(f"  a1 (GHI):              {coefficients['a1']:12.6f}")
    print(f"  a2 (GHI²):             {coefficients['a2']:12.6e}")
    print(f"  a3 (DNI):              {coefficients['a3']:12.6f}")
    print(f"  a4 (GHI × ΔT):         {coefficients['a4']:12.6e}")
    print(f"  a5 (GHI × wind):       {coefficients['a5']:12.6e}")
    print(f"{'='*60}")
    
    return model, coefficients


# =====================================================================
# TRAINING METRICS (on training data only - for monitoring fit quality)
# =====================================================================

def compute_training_metrics(model, X, y, inv_rating=55.0):
    """
    Compute metrics on training data.
    
    NOTE: These metrics are for monitoring training quality only.
    Test performance should be evaluated using evaluate_models.py.
    """
    print(f"\n{'='*60}")
    print("TRAINING SET METRICS (for monitoring only)")
    print(f"{'='*60}")
    
    y_pred_raw = model.predict(X)
    y_pred = np.clip(y_pred_raw, 0, inv_rating)
    
    # Overall metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    # MAPE (excluding near-zero values)
    mape_mask = y > 1.0
    if mape_mask.sum() > 0:
        mape = np.mean(np.abs((y[mape_mask] - y_pred[mape_mask]) / y[mape_mask])) * 100
    else:
        mape = np.nan
    
    # Energy comparison
    total_actual = y.sum()
    total_predicted = y_pred.sum()
    energy_error_pct = abs(total_actual - total_predicted) / total_actual * 100 if total_actual > 0 else 0
    
    print(f"\nFit Quality (training data):")
    print(f"  R² Score:              {r2:.4f}  {'✓' if r2 > 0.95 else '⚠'}")
    print(f"  MAE:                   {mae:.2f} kW")
    print(f"  RMSE:                  {rmse:.2f} kW")
    print(f"  MAPE (daytime):        {mape:.1f}%")
    print(f"  Energy Error:          {energy_error_pct:.2f}%")
    print(f"\n⚠️  These are TRAINING metrics.")
    print(f"   For true performance, run evaluate_models.py on TEST data.")
    print(f"{'='*60}")
    
    metrics = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'energy_error_pct': energy_error_pct,
        'n_samples': len(y)
    }
    
    return y_pred, metrics


# =====================================================================
# OUTPUT GENERATION
# =====================================================================

def save_coefficients(coefficients, metrics, output_dir):
    """Save coefficients in multiple formats."""
    
    print(f"\n{'='*60}")
    print("SAVING COEFFICIENTS")
    print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON format (for programmatic use)
    json_path = output_dir / "surrogate_coefficients.json"
    output_data = {
        "coefficients": coefficients,
        "training_metrics": metrics,
        "feature_order": ["ghi", "ghi_squared", "dni", "ghi_temp", "ghi_wind"],
        "equation": "P_AC = a0 + a1*GHI + a2*GHI² + a3*DNI + a4*(GHI×ΔT) + a5*(GHI×wind)"
    }
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved: {json_path}")
    
    # 2. JavaScript format (for ThingsBoard)
    js_path = output_dir / "js_coefficients.txt"
    js_code = f"""// FITTED REGRESSION COEFFICIENTS
// Training samples: {metrics['n_samples']}
// Training R2: {metrics['r2']:.4f}
// Training MAE: {metrics['mae']:.2f} kW

const REGRESSION_COEFFICIENTS = {{
  a0: {coefficients['a0']:.6f},  // Intercept
  a1: {coefficients['a1']:.6f},  // Linear GHI term
  a2: {coefficients['a2']:.6e},  // Quadratic GHI term
  a3: {coefficients['a3']:.6f},  // DNI contribution
  a4: {coefficients['a4']:.6e},  // Temperature interaction (GHI x deltaT)
  a5: {coefficients['a5']:.6e}   // Wind interaction (GHI x wind)
}};

// Regression equation:
// P_AC = a0 + a1*GHI + a2*GHI^2 + a3*DNI + a4*(GHI*deltaT) + a5*(GHI*wind)
// Then: clip(P_AC, 0, INV_RATING_KW)
"""
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_code)
    print(f"  Saved: {js_path}")
    
    # 3. Print JavaScript code
    print(f"\n{'='*60}")
    print("JAVASCRIPT COEFFICIENTS (COPY-PASTE READY)")
    print(f"{'='*60}")
    print(js_code)
    
    return json_path, js_path


def save_training_plot(df, y_pred, metrics, output_dir):
    """Save training fit visualization."""
    
    output_dir = Path(output_dir)
    y = df['ac_power_kw'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Actual vs Predicted
    ax = axes[0]
    ax.scatter(y, y_pred, alpha=0.3, s=10)
    ax.plot([0, 55], [0, 55], 'r--', linewidth=2, label='Perfect Fit')
    ax.set_xlabel('Actual Power (kW)', fontsize=12)
    ax.set_ylabel('Predicted Power (kW)', fontsize=12)
    ax.set_title(f'Training Fit (R² = {metrics["r2"]:.4f})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    
    # Plot 2: Residuals
    ax = axes[1]
    residuals = y - y_pred
    ax.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (kW)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Residual Distribution (MAE = {metrics["mae"]:.2f} kW)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "training_fit.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    
    plt.close()


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train surrogate model on training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fit_surrogate.py --train-data ../../data/train_data.csv
  python fit_surrogate.py --legacy  # Use old method (not recommended)
        """
    )
    
    parser.add_argument(
        "--train-data",
        default=None,
        help="Path to training data CSV from prepare_training_data.py"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy data loading (for backwards compatibility)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ../../output)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PV SURROGATE MODEL - COEFFICIENT FITTING")
    print("=" * 60)
    
    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    
    # Load data
    if args.train_data:
        df = load_training_data(args.train_data)
    elif args.legacy:
        df = load_legacy_data()
    else:
        # Default: look for train_data.csv
        default_train_path = DATA_DIR / "train_data.csv"
        if default_train_path.exists():
            df = load_training_data(default_train_path)
        else:
            print("\n⚠️  No training data found!")
            print(f"   Expected: {default_train_path}")
            print("\n   Run prepare_training_data.py first, or use --legacy flag.")
            sys.exit(1)
    
    # Engineer features
    X, y, df = engineer_features(df)
    
    # Fit model
    model, coefficients = fit_model(X, y)
    
    # Compute training metrics
    y_pred, metrics = compute_training_metrics(model, X, y)
    
    # Save outputs
    save_coefficients(coefficients, metrics, output_dir)
    
    if not args.no_plot:
        save_training_plot(df, y_pred, metrics, output_dir)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Run evaluate_models.py on TEST data (not training data)")
    print(f"  2. Copy coefficients to surrogate_model.js")
    print(f"  3. Deploy to ThingsBoard")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
