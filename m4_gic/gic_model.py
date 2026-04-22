"""
ICARUS-X — M4 GIC: Empirical GIC Model

Estimates Geomagnetically Induced Currents (GIC) from Kp index
using empirical power-law relationship from published literature.

GIC ∝ 10^(0.28*Kp - 0.90)  [A/km, approximate for high-latitude grids]

Inputs:  Kp predicted value
Outputs: GIC estimate in A/km
"""

import numpy as np
from typing import Dict
from loguru import logger


# ── Empirical model parameters ───────────────────────────
# Based on: Pulkkinen et al. (2012), Viljanen et al. (2006)
A_COEFF = 0.28
B_COEFF = -0.90
BASE = 10.0


def kp_to_gic(kp: float) -> float:
    """Convert Kp index to GIC estimate using empirical formula."""
    kp_clipped = np.clip(kp, 0, 9)
    gic = BASE ** (A_COEFF * kp_clipped + B_COEFF)
    return round(float(gic), 2)


def kp_to_gic_batch(kp_array: np.ndarray) -> np.ndarray:
    """Vectorized Kp → GIC conversion."""
    kp_clipped = np.clip(kp_array, 0, 9)
    return BASE ** (A_COEFF * kp_clipped + B_COEFF)


def gic_to_risk_tier(gic: float) -> Dict[str, str]:
    """Map GIC value to risk tier with associated color."""
    if gic < 1.0:
        return {"tier": "LOW", "color": "#4CAF50"}       # green
    elif gic < 5.0:
        return {"tier": "MEDIUM", "color": "#FFA500"}     # orange
    elif gic < 15.0:
        return {"tier": "HIGH", "color": "#FF5722"}       # red-orange
    else:
        return {"tier": "CRITICAL", "color": "#D50000"}   # deep red


# ── Reference values for validation ─────────────────────
# Sept 2017 storm: Kp=8 → GIC ≈ 18 A/km (observed ~15-20 in Finland)
REFERENCE_TABLE = {
    0: 0.13, 1: 0.24, 2: 0.46, 3: 0.87,
    4: 1.66, 5: 3.16, 6: 6.03, 7: 11.48,
    8: 21.88, 9: 41.69,
}


if __name__ == "__main__":
    print("Kp → GIC Reference Table:")
    print(f"{'Kp':>4} {'GIC (A/km)':>12} {'Risk':>10}")
    print("-" * 30)
    for kp in range(10):
        gic = kp_to_gic(kp)
        risk = gic_to_risk_tier(gic)
        print(f"{kp:4d} {gic:12.2f} {risk['tier']:>10}")
