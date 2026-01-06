# Model Benchmark Report

Generated: 2026-01-05 17:15:42

## Summary

Comparison of three PV prediction models against actual Meteocontrol generation data.

**Comparison Period:** December 11-15, 2025 (5 complete days)
**Timezone:** Asia/Colombo (UTC+5:30)

## Results

| Model | MAE (kWh) | MAPE (%) | Bias (%) | Total Error (%) | Avg Time (ms) |
|-------|-----------|----------|----------|-----------------|---------------|
| Python Physics | 32.7 | 11.4 | 2.8 | 3.7 | 0.421 |
| JS Physics | 41.5 | 14.5 | 11.5 | 12.4 | 0.018 |
| JS Surrogate | 28.4 | 10.0 | 2.4 | 3.2 | 0.005 |

## Daily Energy Comparison (kWh)

| Date | Actual | Python Physics | JS Physics | JS Surrogate |
|------|--------|----------------|------------|--------------|
| 2025-12-11 | 251.7 | 224.5 | 233.3 | 232.6 |
| 2025-12-12 | 283.3 | 263.4 | 301.9 | 265.8 |
| 2025-12-13 | 233.0 | 224.2 | 250.0 | 221.0 |
| 2025-12-14 | 362.1 | 423.4 | 444.6 | 413.7 |
| 2025-12-15 | 250.4 | 296.8 | 321.5 | 292.1 |

## Notes

- **Python Physics**: Full pvlib model with Perez POA transposition, SAPM thermal model
- **JS Physics**: Simplified Perez approximation, designed for ThingsBoard
- **JS Surrogate**: Regression-based model (coefficients may need training)
- All timestamps converted to Asia/Colombo before daily aggregation
- Only complete days (Dec 11-15) included in comparison
