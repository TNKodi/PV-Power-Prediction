"""
Model Validation Script

Compares model predictions against actual Meteocontrol generation data.
Uses timezone utilities for consistent timestamp handling.

Usage:
    python validate_model.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Timezone utilities for consistent handling
from timezone_utils import load_meteocontrol, to_local, LOCAL_TZ

# ============================================================
# FILE PATHS (EDIT IF NEEDED)
# ============================================================

# Path: scripts/shared/ -> ../../output, ../../data
PROJECT_DIR = Path(__file__).parent.parent.parent
MODEL_CSV = PROJECT_DIR / "output" / "pv_generation.csv"
ACTUAL_CSV = PROJECT_DIR / "data" / "meteocontrol_actual.csv"

# ============================================================
# 1) LOAD MODEL OUTPUT (HOURLY kW)
# ============================================================

print("\n" + "=" * 70)
print("MODEL VALIDATION")
print("=" * 70)

print(f"\nLoading model output: {MODEL_CSV}")
model = pd.read_csv(MODEL_CSV, parse_dates=["period_end"])
model = model.set_index("period_end").sort_index()

# Sanity check
assert "AC_kW" in model.columns, "AC_kW column missing"

# Ensure timezone-aware
if model.index.tz is None:
    print("  Warning: Model output has no timezone, assuming Asia/Colombo")
    model.index = model.index.tz_localize(LOCAL_TZ)
else:
    model = to_local(model)

print(f"  Loaded {len(model)} hourly records")
print(f"  Timezone: {model.index.tz}")
print(f"  Date range: {model.index.min()} to {model.index.max()}")

# ------------------------------------------------------------
# Convert power → energy (kWh)
# ------------------------------------------------------------
# Infer timestep (hours)
dt_hours = (model.index[1] - model.index[0]).total_seconds() / 3600

model["Energy_kWh"] = model["AC_kW"] * dt_hours

# Aggregate to DAILY energy (keep timezone-aware)
model_daily = model["Energy_kWh"].resample("D").sum()

# ============================================================
# 2) LOAD METEOCONTROL DATA (DAILY kWh)
# ============================================================

print(f"\nLoading actual data: {ACTUAL_CSV}")
actual = load_meteocontrol(ACTUAL_CSV)

print(f"  Loaded {len(actual)} daily records")
print(f"  Timezone: {actual.index.tz}")
print(f"  Date range: {actual.index.min()} to {actual.index.max()}")

actual_daily = actual["actual_kwh"]

# ============================================================
# 3) ALIGN COMMON DATE RANGE
# ============================================================

# Find common date range (both are now timezone-aware in Asia/Colombo)
common_start = max(model_daily.index.min(), actual_daily.index.min())
common_end = min(model_daily.index.max(), actual_daily.index.max())

print(f"\nCommon date range: {common_start.date()} to {common_end.date()}")

model_daily = model_daily.loc[common_start:common_end]
actual_daily = actual_daily.loc[common_start:common_end]

comparison = pd.DataFrame({
    "Model_kWh": model_daily,
    "Actual_kWh": actual_daily
}).dropna()

print(f"Comparing {len(comparison)} days")

# ============================================================
# 4) ERROR METRICS
# ============================================================

comparison["Error_kWh"] = comparison["Model_kWh"] - comparison["Actual_kWh"]
comparison["Error_%"] = comparison["Error_kWh"] / comparison["Actual_kWh"] * 100

MAE = comparison["Error_kWh"].abs().mean()
MAPE = comparison["Error_%"].abs().mean()
BIAS = comparison["Error_%"].mean()

# ============================================================
# 5) PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DAILY ENERGY COMPARISON (MODEL vs ACTUAL)")
print("=" * 70)

# Format for display (remove timezone for cleaner output)
display_comparison = comparison.copy()
display_comparison.index = display_comparison.index.strftime("%Y-%m-%d")
print(display_comparison.round(2))

print("\n" + "=" * 70)
print("ERROR METRICS")
print("=" * 70)
print(f"Mean Absolute Error (kWh): {MAE:.2f}")
print(f"Mean Absolute Percentage Error (%): {MAPE:.2f}")
print(f"Bias (%): {BIAS:.2f}")
print("=" * 70)

# Total energy comparison
total_model = comparison["Model_kWh"].sum()
total_actual = comparison["Actual_kWh"].sum()
total_error_pct = (total_model - total_actual) / total_actual * 100

print(f"\nTotal Energy:")
print(f"  Model:  {total_model:.1f} kWh")
print(f"  Actual: {total_actual:.1f} kWh")
print(f"  Error:  {total_error_pct:+.1f}%")
print("=" * 70)

# ============================================================
# 6) PLOTS
# ============================================================

# --- Plot 1: Daily Energy Comparison ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax = axes[0]
dates = comparison.index
ax.bar(range(len(dates)), comparison["Actual_kWh"], width=0.4, 
       label="Actual (Meteocontrol)", alpha=0.7, align="edge")
ax.bar([x + 0.4 for x in range(len(dates))], comparison["Model_kWh"], width=0.4,
       label="Model (Physics)", alpha=0.7, align="edge")
ax.set_xticks(range(len(dates)))
ax.set_xticklabels([d.strftime("%b %d") for d in dates], rotation=45)
ax.set_ylabel("Daily Energy (kWh)")
ax.set_title("Daily Energy Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot 2: Daily Percentage Error ---
ax = axes[1]
colors = ["green" if e < 0 else "red" for e in comparison["Error_%"]]
ax.bar(range(len(dates)), comparison["Error_%"], color=colors, alpha=0.7)
ax.axhline(0, color="black", linewidth=0.5)
ax.axhline(BIAS, color="blue", linestyle="--", label=f"Bias: {BIAS:.1f}%")
ax.set_xticks(range(len(dates)))
ax.set_xticklabels([d.strftime("%b %d") for d in dates], rotation=45)
ax.set_ylabel("Error (%)")
ax.set_title(f"Daily Model Error (MAPE: {MAPE:.1f}%)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
output_path = PROJECT_DIR / "output" / "validation_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {output_path}")

plt.show()
