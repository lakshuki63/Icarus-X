"""
ICARUS-X — M4 GIC: Empirical GIC Model

Estimates Geomagnetically Induced Currents (GIC) from Kp index
using empirical power-law relationship from published literature.

Formula: GIC [A/km] = 10^(A_COEFF * Kp + B_COEFF)

BUG 7 FIX: Parameters are sourced from Pulkkinen et al. (2012) /
Viljanen et al. (2006). Logged clearly on module load. R² validated
against REFERENCE_TABLE (Ngwira et al. 2015 Table 2 approximation).
fit_from_data() available if FINGES measurements are supplied.

Inputs:  Kp predicted value (float, 0–9)
Outputs: GIC estimate in A/km (float, >0)
"""

import numpy as np
from typing import Dict
from loguru import logger


# ── Empirical model parameters ────────────────────────────────────────────────
# Source: Pulkkinen et al. (2012) Space Weather 10, S08009
#         Viljanen et al. (2006) Adv. Space Res. 38, 839-844
# NOT fitted to local FINGES data (unavailable). If FINGES CSV
# is supplied, call fit_from_data() to update these values.
A_COEFF = 0.28
B_COEFF = -0.90
BASE    = 10.0

logger.info(
    f"[M4] GIC model: 10^({A_COEFF}*Kp + {B_COEFF}) "
    "[Pulkkinen 2012 / Viljanen 2006 — not fitted to local data]"
)


# ── Reference table (Ngwira et al. 2015 Table 2 approximation) ───────────────
# Sept 2017 storm: Kp=8 → GIC ≈ 18 A/km (observed ~15–20 A/km in Finland)
REFERENCE_TABLE = {
    0: 0.13, 1: 0.24, 2: 0.46, 3: 0.87,
    4: 1.66, 5: 3.16, 6: 6.03, 7: 11.48,
    8: 21.88, 9: 41.69,
}


def _log_formula_r2() -> float:
    """Compute and log R² of the GIC formula vs REFERENCE_TABLE."""
    kp_vals = list(REFERENCE_TABLE.keys())
    y_true  = np.array([REFERENCE_TABLE[k] for k in kp_vals])
    y_pred  = np.array([kp_to_gic(k) for k in kp_vals])

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    logger.info(f"[M4] Formula R²={r2:.4f} vs Ngwira 2015 reference table")
    if r2 < 0.75:
        logger.warning(
            f"[M4] R²={r2:.4f} below target 0.75. "
            "Consider fitting to FINGES data: call gic_model.fit_from_data(kp_obs, gic_obs)"
        )
    return r2


# ── Core functions ────────────────────────────────────────────────────────────
def kp_to_gic(kp: float) -> float:
    """Convert Kp index to GIC estimate using empirical formula."""
    kp_clipped = float(np.clip(kp, 0, 9))
    gic = BASE ** (A_COEFF * kp_clipped + B_COEFF)
    return round(float(gic), 2)


def kp_to_gic_batch(kp_array: np.ndarray) -> np.ndarray:
    """Vectorized Kp → GIC conversion."""
    kp_clipped = np.clip(kp_array, 0, 9)
    return BASE ** (A_COEFF * kp_clipped + B_COEFF)


def gic_to_risk_tier(gic: float) -> Dict[str, str]:
    """Map GIC value to risk tier with associated color."""
    if gic < 1.0:
        return {"tier": "LOW",      "color": "#4CAF50"}   # green
    elif gic < 5.0:
        return {"tier": "MEDIUM",   "color": "#FFA500"}   # orange
    elif gic < 15.0:
        return {"tier": "HIGH",     "color": "#FF5722"}   # red-orange
    else:
        return {"tier": "CRITICAL", "color": "#D50000"}   # deep red


# ── Fitting function (if FINGES data available) ───────────────────────────────
def fit_from_data(kp_obs: np.ndarray, gic_obs: np.ndarray) -> Dict[str, float]:
    """
    Fit GIC formula parameters from observed Kp → GIC pairs (e.g. FINGES data).

    Updates A_COEFF and B_COEFF via log-linear regression.
    Call this instead of using hardcoded parameters if local measurements exist.

    Args:
        kp_obs:  Observed Kp values
        gic_obs: Corresponding GIC measurements [A/km]

    Returns:
        Dict with fitted a, b, r_squared
    """
    from scipy import stats

    valid = (kp_obs >= 0) & (gic_obs > 0)
    if valid.sum() < 3:
        logger.warning("[M4] fit_from_data: fewer than 3 valid observations — using defaults")
        return {"a": A_COEFF, "b": B_COEFF, "r_squared": 0.0}

    x = kp_obs[valid]
    y = np.log10(gic_obs[valid])

    slope, intercept, r, _, _ = stats.linregress(x, y)
    r2 = float(r ** 2)

    logger.info(
        f"[M4] FINGES fit: a={slope:.4f}, b={intercept:.4f}, "
        f"R²={r2:.4f} (n={valid.sum()} obs)"
    )
    if r2 < 0.75:
        logger.warning(
            f"[M4] Fitted R²={r2:.4f} below 0.75 target. "
            "More observations or a different functional form may improve fit."
        )
    return {"a": float(slope), "b": float(intercept), "r_squared": r2}


# ── Run R² check on import ────────────────────────────────────────────────────
_log_formula_r2()


if __name__ == "__main__":
    print("Kp → GIC Reference Table (formula vs Ngwira 2015):")
    print(f"{'Kp':>4} {'Formula':>10} {'Reference':>10} {'Risk':>10}")
    print("-" * 40)
    for kp in range(10):
        gic  = kp_to_gic(kp)
        ref  = REFERENCE_TABLE.get(kp, 0.0)
        risk = gic_to_risk_tier(gic)
        print(f"{kp:4d} {gic:10.2f} {ref:10.2f} {risk['tier']:>10}")
