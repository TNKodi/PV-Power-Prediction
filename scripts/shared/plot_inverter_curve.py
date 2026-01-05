# =====================================================================
# HELPER: VERIFY & PRINT HARDCODED INVERTER EFFICIENCY ARRAYS (CORRECTED)
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt

P_RATED_KW = 50.0

# Digitized datasheet points (RAW, UNSORTED, AS GIVEN)
x_raw = np.array([
    4.868913857677905,
    19.975031210986273,
    40.07490636704121,
    60.049937578027475,
    80.14981273408239,
    100.0,
    90.38701622971286,
    85.14357053682896,
    95.00624219725344,
    69.91260923845194,
    75.03121098626717,
    64.91885143570536,
    49.81273408239702,
    55.0561797752809,
    44.569288389513105,
    29.83770287141074,
    34.831460674157306,
    24.469413233458184,
    9.488139825218484,
    9.987515605493135,
    14.357053682896387
])

y_raw = np.array([
    95.23783783783784,
    98.01621621621622,
    98.27567567567569,
    98.22162162162162,
    98.05945945945946,
    97.87567567567568,
    97.97297297297298,
    98.01621621621622,
    97.92972972972973,
    98.14594594594595,
    98.1027027027027,
    98.1891891891892,
    98.27567567567569,
    98.25405405405405,
    98.2864864864865,
    98.22162162162162,
    98.25405405405405,
    98.16756756756757,
    97.15135135135135,
    97.27027027027027,
    97.72432432432433
])

# =====================================================================
# EXACT conversion — nothing else
# =====================================================================
kw  = (x_raw / 100.0) * P_RATED_KW
eta = y_raw / 100.0

# =====================================================================
# SORT PAIRS ONCE — THIS IS THE FIX
# =====================================================================
sort_idx = np.argsort(kw)
inverter_eff_curve_kw  = kw[sort_idx]
inverter_eff_curve_eta = eta[sort_idx]

# =====================================================================
# SANITY CHECKS (DO NOT REMOVE IN PROD)
# =====================================================================
assert np.all(np.diff(inverter_eff_curve_kw) >= 0), "kW array not sorted!"
assert len(inverter_eff_curve_kw) == len(inverter_eff_curve_eta)

# =====================================================================
# PLOT FOR VERIFICATION
# =====================================================================
plt.figure(figsize=(8, 5))

plt.scatter(
    inverter_eff_curve_kw,
    inverter_eff_curve_eta * 100,
    color="red",
    zorder=3,
    label="Digitized points"
)

plt.plot(
    inverter_eff_curve_kw,
    inverter_eff_curve_eta * 100,
    linestyle="--",
    alpha=0.7,
    label="Efficiency curve (sorted)"
)

plt.xlabel("DC Input Power (kW)")
plt.ylabel("Inverter Efficiency (%)")
plt.title("Inverter Efficiency Curve — VERIFIED & SORTED")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================================
# PRINT IN COPY-PASTE-SAFE FORM (SORTED!)
# =====================================================================
print("\n# ---- HARD-CODED INVERTER EFFICIENCY CURVE ----")
print("inverter_eff_curve_kw = np.array([")
print(", ".join(f"{v:.2f}" for v in inverter_eff_curve_kw))
print("])\n")

print("inverter_eff_curve_eta = np.array([")
print(", ".join(f"{v:.5f}" for v in inverter_eff_curve_eta))
print("])")
print("# --------------------------------------------\n")
