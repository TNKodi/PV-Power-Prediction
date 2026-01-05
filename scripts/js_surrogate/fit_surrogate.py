"""
PV Surrogate Model - Local Coefficient Fitting Script

This script:
1. Reads your weather CSV + Python output CSV
2. Fits regression coefficients using sklearn
3. Validates the model
4. Outputs JS-ready coefficients

Run this locally on your machine.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# =====================================================================
# STEP 1: LOAD DATA
# =====================================================================

print("="*60)
print("PV SURROGATE MODEL - COEFFICIENT FITTING")
print("="*60)

# Load weather inputs
# Path: scripts/js_surrogate/ -> ../../data/
input_csv = r"..\..\data\solcast_irradiance.csv"
output_csv = r"..\..\output\pv_generation.csv"

print(f"\nLoading input data: {input_csv}")
df_input = pd.read_csv(input_csv)

print(f"Loading output data: {output_csv}")
df_output = pd.read_csv(output_csv)

# =====================================================================
# TIMESTAMP NORMALIZATION (CRITICAL)
# =====================================================================

# Parse timestamps
df_input['period_end'] = pd.to_datetime(df_input['period_end'], utc=True)
df_output['period_end'] = pd.to_datetime(df_output['period_end'])

# Convert PV output (Asia/Colombo) → UTC
df_output['period_end'] = df_output['period_end'].dt.tz_convert('UTC')

# PV output timestamps are mid-interval (HH:30) → snap to period_end
df_output['period_end'] = df_output['period_end'].dt.ceil('H')

# Safety check: enforce hourly timestamps
if df_output['period_end'].dt.minute.any():
    raise RuntimeError(
        "Non-hourly timestamps detected after alignment. "
        "Check PV output timestamp semantics."
    )

# Optional: debug print (can remove later)
print("\nAligned timestamps (weather):")
print(df_input['period_end'].head(3))

print("\nAligned timestamps (PV output):")
print(df_output['period_end'].head(3))

# =====================================================================
# MERGE DATASETS
# =====================================================================

print("\nMerging datasets...")
df = pd.merge(
    df_input,
    df_output,
    on='period_end',
    how='inner'
)

# Rename wind_speed column if needed
if 'wind_speed_10m' in df.columns:
    df['wind_speed'] = df['wind_speed_10m']

print(f"✓ Merged dataset: {len(df)} samples")
print(f"  Columns: {list(df.columns)}")

# =====================================================================
# STEP 2: FEATURE ENGINEERING
# =====================================================================

print("\nEngineering features...")

# Temperature deviation from STC
df['delta_T'] = df['air_temp'] - 25

# Regression features
df['ghi_squared'] = df['ghi'] ** 2
df['ghi_temp'] = df['ghi'] * df['delta_T']
df['ghi_wind'] = df['ghi'] * df['wind_speed']

# Feature matrix X
X = df[['ghi', 'ghi_squared', 'dni', 'ghi_temp', 'ghi_wind']].values

# Target variable y
y = df['AC_kW'].values

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Target vector shape: {y.shape}")

# =====================================================================
# STEP 3: FIT REGRESSION MODEL
# =====================================================================

print("\nFitting regression model...")

model = LinearRegression(fit_intercept=True)
model.fit(X, y)

# Extract coefficients
a0 = model.intercept_
a1, a2, a3, a4, a5 = model.coef_

print("✓ Regression complete")
print(f"\n{'='*60}")
print("FITTED COEFFICIENTS")
print(f"{'='*60}")
print(f"  a0 (intercept):        {a0:12.6f}")
print(f"  a1 (GHI):              {a1:12.6f}")
print(f"  a2 (GHI²):             {a2:12.6e}")
print(f"  a3 (DNI):              {a3:12.6f}")
print(f"  a4 (GHI × ΔT):         {a4:12.6e}")
print(f"  a5 (GHI × wind):       {a5:12.6e}")
print(f"{'='*60}")

# =====================================================================
# STEP 4: MAKE PREDICTIONS
# =====================================================================

print("\nGenerating predictions...")

y_pred_raw = model.predict(X)

# Apply physical constraints (matching JS implementation)
INV_RATING_KW = 55.0
y_pred = np.clip(y_pred_raw, 0, INV_RATING_KW)

# Add to dataframe
df['predicted_kW'] = y_pred
df['error_kW'] = np.abs(y - y_pred)
df['error_pct'] = np.where(y > 1.0, (df['error_kW'] / y) * 100, np.nan)

# =====================================================================
# STEP 5: VALIDATION METRICS
# =====================================================================

print(f"\n{'='*60}")
print("VALIDATION METRICS")
print(f"{'='*60}")

# Overall metrics
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# MAPE (excluding near-zero values)
mape_mask = y > 1.0
mape = np.mean(np.abs((y[mape_mask] - y_pred[mape_mask]) / y[mape_mask])) * 100

# Daily/period energy
total_actual_kwh = y.sum()
total_predicted_kwh = y_pred.sum()
energy_error_pct = abs(total_actual_kwh - total_predicted_kwh) / total_actual_kwh * 100

print(f"\nOverall Performance:")
print(f"  R² Score:              {r2:.4f}  {'✓ Excellent' if r2 > 0.95 else '⚠ Review'}")
print(f"  MAE:                   {mae:.2f} kW  ({mae/55*100:.1f}% of rated)")
print(f"  RMSE:                  {rmse:.2f} kW")
print(f"  MAPE (daytime only):   {mape:.1f}%  {'✓ Good' if mape < 10 else '⚠ High'}")

print(f"\nEnergy Comparison (Total Period):")
print(f"  Actual (Python):       {total_actual_kwh:.1f} kWh")
print(f"  Predicted (Surrogate): {total_predicted_kwh:.1f} kWh")
print(f"  Absolute Error:        {abs(total_actual_kwh - total_predicted_kwh):.1f} kWh")
print(f"  Relative Error:        {energy_error_pct:.2f}%  {'✓ Target met' if energy_error_pct < 5 else '⚠ High'}")

print(f"\nSample Statistics:")
print(f"  Total samples:         {len(df)}")
print(f"  Daytime samples:       {np.sum(y > 1)} (GHI > ~50 W/m²)")
print(f"  Max actual power:      {y.max():.1f} kW")
print(f"  Max predicted power:   {y_pred.max():.1f} kW")

print(f"{'='*60}")

# =====================================================================
# STEP 6: EXPORT JAVASCRIPT COEFFICIENTS
# =====================================================================

print(f"\n{'='*60}")
print("JAVASCRIPT COEFFICIENTS (COPY-PASTE READY)")
print(f"{'='*60}\n")

js_code = f"""// FITTED REGRESSION COEFFICIENTS FOR YOUR PLANT
// Generated from {len(df)} hourly samples
// Validation: R²={r2:.4f}, MAE={mae:.2f} kW, Energy Error={energy_error_pct:.2f}%

const REGRESSION_COEFFICIENTS = {{
  a0: {a0:.6f},  // Intercept
  a1: {a1:.6f},  // Linear GHI term
  a2: {a2:.6e},  // Quadratic GHI term
  a3: {a3:.6f},  // DNI contribution
  a4: {a4:.6e},  // Temperature interaction (GHI × ΔT)
  a5: {a5:.6e}   // Wind interaction (GHI × wind)
}};

// Regression equation:
// P_AC = {a0:.6f}
//      + {a1:.6f} × GHI
//      + {a2:.6e} × GHI²
//      + {a3:.6f} × DNI
//      + {a4:.6e} × (GHI × ΔT)
//      + {a5:.6e} × (GHI × wind_speed)
"""

print(js_code)

# Save to file
with open(r"..\..\output\js_coefficients.txt", "w", encoding="utf-8") as f:
    f.write(js_code)

print(f"✓ Coefficients saved to: ../../output/js_coefficients.txt")

# =====================================================================
# STEP 7: SAVE VALIDATION RESULTS
# =====================================================================

print(f"\n{'='*60}")
print("SAVING VALIDATION OUTPUTS")
print(f"{'='*60}")

# Save detailed predictions
output_columns = [
    'period_end', 'ghi', 'dni', 'air_temp', 'wind_speed',
    'AC_kW', 'predicted_kW', 'error_kW', 'error_pct'
]
df[output_columns].to_csv(r"..\..\output\validation_results.csv", index=False)
print(f"✓ Validation results saved to: ../../output/validation_results.csv")

# =====================================================================
# STEP 8: GENERATE PLOTS (OPTIONAL)
# =====================================================================

print(f"\nGenerating validation plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs Predicted (Scatter)
ax = axes[0, 0]
ax.scatter(y, y_pred, alpha=0.6, s=30)
ax.plot([0, 55], [0, 55], 'r--', linewidth=2, label='Perfect Fit')
ax.set_xlabel('Actual Power (kW)', fontsize=12)
ax.set_ylabel('Predicted Power (kW)', fontsize=12)
ax.set_title(f'Actual vs Predicted (R² = {r2:.4f})', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Time Series
ax = axes[0, 1]
hours = np.arange(len(y))
ax.plot(hours, y, 'b-', linewidth=2, label='Actual (Python)', alpha=0.7)
ax.plot(hours, y_pred, 'r--', linewidth=2, label='Predicted (Surrogate)', alpha=0.7)
ax.set_xlabel('Hour Index', fontsize=12)
ax.set_ylabel('AC Power (kW)', fontsize=12)
ax.set_title('Hourly Power Timeline', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Errors vs GHI
ax = axes[1, 0]
ax.scatter(df['ghi'], df['error_kW'], alpha=0.6, s=30, c='green')
ax.set_xlabel('GHI (W/m²)', fontsize=12)
ax.set_ylabel('Absolute Error (kW)', fontsize=12)
ax.set_title(f'Prediction Errors vs Irradiance (MAE = {mae:.2f} kW)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='r', linestyle='--', linewidth=1)

# Plot 4: Error Distribution
ax = axes[1, 1]
ax.hist(df['error_kW'], bins=30, color='purple', alpha=0.7, edgecolor='black')
ax.set_xlabel('Absolute Error (kW)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'Error Distribution (Mean = {mae:.2f} kW)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'..\..\output\validation_plots.png', dpi=150, bbox_inches='tight')
print(f"✓ Validation plots saved to: ../../output/validation_plots.png")

plt.show()

print(f"\n{'='*60}")
print("COEFFICIENT FITTING COMPLETE")
print(f"{'='*60}")
print("\nNext steps:")
print("1. Review validation_plots.png")
print("2. Check validation_results.csv for detailed errors")
print("3. Copy coefficients from js_coefficients.txt")
print("4. Paste into your JavaScript surrogate model")
print("5. Deploy to ThingsBoard")
print(f"{'='*60}\n")