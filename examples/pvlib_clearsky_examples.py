#!/usr/bin/env python3
"""
Quick Start Example: Using pvlib Clear Sky Models
==================================================

This script demonstrates how to use the updated physics_model.py
with pvlib clear sky models for GHI, DNI, DHI data.
"""

import subprocess
import os
from pathlib import Path

# Change to the physics model directory
script_dir = Path(__file__).parent.parent / "scripts" / "python_physics"
os.chdir(script_dir)

print("="*70)
print("PVLIB CLEAR SKY MODEL - QUICK START EXAMPLES")
print("="*70)

# Example 1: Basic usage with default Ineichen model
print("\n1️⃣  Example 1: Basic usage (Ineichen model, 1 week)")
print("-" * 70)
cmd1 = [
    "python", "physics_model.py",
    "--source", "pvlib_clearsky",
    "--start", "20260110",
    "--end", "20260117",
    "--output", "../../output/clearsky_ineichen.csv"
]
print("Command:", " ".join(cmd1))
print("\nThis will generate:")
print("  - Clear sky GHI, DNI, DHI using Ineichen model")
print("  - Hourly data for Jan 10-17, 2026")
print("  - Full PV generation calculation with all physics")
print("  - Output: output/clearsky_ineichen.csv")

# Example 2: Simplified Solis model
print("\n\n2️⃣  Example 2: Simplified Solis model")
print("-" * 70)
cmd2 = [
    "python", "physics_model.py",
    "--source", "pvlib_clearsky",
    "--start", "20260110",
    "--end", "20260117",
    "--clearsky-model", "simplified_solis",
    "--output", "../../output/clearsky_solis.csv"
]
print("Command:", " ".join(cmd2))
print("\nThis uses Simplified Solis model:")
print("  - Faster than Ineichen")
print("  - Good accuracy for most applications")
print("  - Suitable for sensitivity analysis")

# Example 3: Haurwitz model (fastest)
print("\n\n3️⃣  Example 3: Haurwitz model (fastest)")
print("-" * 70)
cmd3 = [
    "python", "physics_model.py",
    "--source", "pvlib_clearsky",
    "--start", "20260110",
    "--end", "20260117",
    "--clearsky-model", "haurwitz",
    "--output", "../../output/clearsky_haurwitz.csv"
]
print("Command:", " ".join(cmd3))
print("\nThis uses Haurwitz model:")
print("  - Fastest computation")
print("  - Simple polynomial approximation")
print("  - Good for quick estimates")

# Example 4: High-resolution data (15-minute intervals)
print("\n\n4️⃣  Example 4: High-resolution data (15-minute intervals)")
print("-" * 70)
cmd4 = [
    "python", "physics_model.py",
    "--source", "pvlib_clearsky",
    "--start", "20260110",
    "--end", "20260112",
    "--frequency", "15T",
    "--output", "../../output/clearsky_15min.csv"
]
print("Command:", " ".join(cmd4))
print("\nThis generates 15-minute interval data:")
print("  - Higher temporal resolution")
print("  - Better for intraday analysis")
print("  - Larger output file")

# Example 5: Monthly analysis
print("\n\n5️⃣  Example 5: Full month analysis")
print("-" * 70)
cmd5 = [
    "python", "physics_model.py",
    "--source", "pvlib_clearsky",
    "--start", "20260101",
    "--end", "20260131",
    "--output", "../../output/clearsky_january.csv"
]
print("Command:", " ".join(cmd5))
print("\nThis generates data for entire month:")
print("  - January 2026 (31 days)")
print("  - Useful for monthly performance analysis")
print("  - Compare with actual monthly generation")

# Comparison example
print("\n\n6️⃣  Example 6: Compare clear sky vs. actual data")
print("-" * 70)
print("Step 1 - Generate clear sky baseline:")
print("  python physics_model.py --source pvlib_clearsky \\")
print("    --start 20251201 --end 20251215 \\")
print("    --output ../../output/baseline_clearsky.csv")
print("\nStep 2 - Generate from actual data (NASA POWER):")
print("  python physics_model.py --source nasa_power \\")
print("    --start 20251201 --end 20251215 \\")
print("    --output ../../output/actual_weather.csv")
print("\nStep 3 - Compare results:")
print("  python ../shared/compare_accuracy.py")

print("\n" + "="*70)
print("IMPORTANT NOTES")
print("="*70)
print("""
1. Install pvlib if not already installed:
   pip install pvlib

2. The model uses location parameters from config/plant_config.json:
   - latitude
   - longitude
   - altitude (important for accuracy!)

3. Temperature estimation is simplified - for production forecasting,
   use Solcast API or NASA POWER for real weather data.

4. Clear sky models assume perfect conditions:
   - No clouds
   - Standard atmospheric conditions
   - No aerosol interference
   
5. Use clear sky for:
   ✓ Theoretical maximum generation
   ✓ Performance ratio analysis
   ✓ System health monitoring
   ✓ Quick feasibility studies
   
   DO NOT use for:
   ✗ Actual weather forecasting
   ✗ Day-ahead predictions
   ✗ Energy trading decisions
""")

print("\n" + "="*70)
print("READY TO RUN")
print("="*70)
print("\nTo execute Example 1, run:")
print("  cd", script_dir)
print("  " + " ".join(cmd1))

print("\nFor more examples and details, see:")
print("  docs/pvlib_clearsky_usage.md")
print("  docs/CHANGES_SUMMARY.md")
print("="*70)
