"""
ICARUS-X — M4 GIC: Full Pipeline

kp_to_gic_risk() — called by M5 model_runner with M2 output ONLY.
M3 flare output is merged separately by M5 — NOT by this module.

Converts M2 Kp forecasts → GIC estimates → risk tiers → headline alert.

Inputs:
  m2_output: M2 forecast dict (8 horizons with kp_predicted, CI bounds)
  bz:        Current Bz (nT) — used for physics-informed uncertainty
  speed:     Current solar wind speed (km/s)
  density:   Current proton density (cm⁻³)

Outputs (M4 contract):
  {
    "run_timestamp": str,
    "horizons": [ {horizon_hr, kp_predicted, kp_ci_low, kp_ci_high,
                   gic_mean, gic_p5, gic_p95,
                   risk_tier, risk_tier_worst, risk_color}, ... ],
    "headline_alert": {alert_level, alert_color, peak_horizon_hr,
                       peak_gic_estimate, message, should_email, should_sms},
    "is_stub": bool
  }
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from m4_gic.gic_model import kp_to_gic, gic_to_risk_tier, REFERENCE_TABLE, A_COEFF, B_COEFF
from m4_gic.uncertainty import estimate_gic_uncertainty
from m4_gic.alert_logic import determine_alert_level

# ── Log GIC model parameters on first import ──────────────────────────────────
logger.info(
    f"[M4] GIC empirical model: GIC = 10^({A_COEFF}*Kp + {B_COEFF}) "
    f"[Pulkkinen et al. 2012, Viljanen et al. 2006]"
)

# ── Validate formula against reference table ──────────────────────────────────
def _validate_gic_formula() -> None:
    """
    Validate GIC formula against published reference values.
    Logs R² and max error so training output can be verified.
    Reference: Pulkkinen et al. 2012, Ngwira et al. 2015 Table 2.
    """
    import numpy as np
    kp_vals = list(REFERENCE_TABLE.keys())
    y_true  = np.array([REFERENCE_TABLE[k] for k in kp_vals])
    y_pred  = np.array([kp_to_gic(k) for k in kp_vals])

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    max_err = float(np.max(np.abs(y_true - y_pred)))

    logger.info(f"[M4] GIC formula validation: R²={r2:.4f}, max_error={max_err:.2f} A/km")
    if r2 < 0.75:
        logger.warning(
            f"[M4] R²={r2:.4f} below target 0.75. "
            "Consider fitting to FINGES measurements if available."
        )

_validate_gic_formula()


# ── Main pipeline ─────────────────────────────────────────────────────────────
def kp_to_gic_risk(
    m2_output: Dict[str, Any],
    bz:        float = 0.0,
    speed:     float = 400.0,
    density:   float = 5.0,
) -> Dict[str, Any]:
    """
    Full M4 pipeline: M2 Kp forecast → GIC risk assessment.

    M3 output is NOT accepted here — M5 model_runner merges M3 separately.

    Args:
        m2_output: M2 forecast dict with 'horizons' list
        bz:        Current Bz (nT)
        speed:     Current solar wind speed (km/s)
        density:   Current proton density (cm⁻³)

    Returns:
        M4 output contract dict
    """
    try:
        horizons_in  = m2_output.get("horizons", [])
        if not horizons_in:
            logger.warning("[M4] M2 output has no horizons — using fallback")
            return _stub_gic_risk()

        horizons_out: List[Dict] = []

        for h in horizons_in:
            kp      = float(h.get("kp_predicted", 0))
            kp_low  = float(h.get("kp_ci_low",   max(0.0, kp - 1.0)))
            kp_high = float(h.get("kp_ci_high",  min(9.0, kp + 1.0)))

            # Clip Kp to physical range
            kp      = float(max(0.0, min(9.0, kp)))
            kp_low  = float(max(0.0, min(9.0, kp_low)))
            kp_high = float(max(0.0, min(9.0, kp_high)))

            # GIC with physics-informed uncertainty
            gic_result = estimate_gic_uncertainty(kp, bz, speed, density)

            # Clip GIC to physical range [0, 100] A/km
            gic_mean = float(max(0.0, min(100.0, gic_result["gic_mean"])))
            gic_p5   = float(max(0.0, min(100.0, gic_result["gic_p5"])))
            gic_p95  = float(max(0.0, min(100.0, gic_result["gic_p95"])))

            assert gic_mean >= 0, f"GIC mean must be > 0, got {gic_mean}"

            risk        = gic_to_risk_tier(gic_mean)
            risk_worst  = gic_to_risk_tier(gic_p95)

            horizons_out.append({
                "horizon_hr":      int(h.get("horizon_hr", 0)),
                "kp_predicted":    round(kp,      2),
                "kp_ci_low":       round(kp_low,  2),
                "kp_ci_high":      round(kp_high, 2),
                "gic_mean":        round(gic_mean, 2),
                "gic_p5":          round(gic_p5,   2),
                "gic_p95":         round(gic_p95,  2),
                "risk_tier":       risk["tier"],
                "risk_tier_worst": risk_worst["tier"],
                "risk_color":      risk["color"],
            })

        headline_alert = determine_alert_level(horizons_out)

        result = {
            "run_timestamp":  m2_output.get("run_timestamp", ""),
            "horizons":       horizons_out,
            "headline_alert": headline_alert,
            "is_stub":        m2_output.get("is_stub", False),
        }

        logger.info(
            f"[M4] Alert={headline_alert['alert_level']}, "
            f"Peak GIC={headline_alert['peak_gic_estimate']:.1f} A/km "
            f"@ +{headline_alert['peak_horizon_hr']}h"
        )
        return result

    except Exception as e:
        logger.error(f"[M4] Pipeline error: {e}")
        return _stub_gic_risk()


# ── Fallback ──────────────────────────────────────────────────────────────────
def _stub_gic_risk() -> Dict[str, Any]:
    """
    Safe fallback M4 output with 8 placeholder horizons.
    Returns LOW-risk zeros — NOT fake storm data.
    All 8 horizons present so frontend table never crashes.
    """
    horizons = [
        {
            "horizon_hr":      h,
            "kp_predicted":    0.0,
            "kp_ci_low":       0.0,
            "kp_ci_high":      0.0,
            "gic_mean":        0.13,   # Kp=0 baseline
            "gic_p5":          0.05,
            "gic_p95":         0.30,
            "risk_tier":       "LOW",
            "risk_tier_worst": "LOW",
            "risk_color":      "#4CAF50",
        }
        for h in [3, 6, 9, 12, 15, 18, 21, 24]
    ]
    return {
        "run_timestamp": "",
        "horizons":      horizons,
        "headline_alert": {
            "alert_level":       "LOW",
            "alert_color":       "#4CAF50",
            "peak_horizon_hr":   0,
            "peak_gic_estimate": 0.13,
            "message":           "System initializing — no forecast available.",
            "should_email":      False,
            "should_sms":        False,
        },
        "is_stub": True,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from m2_predictor.infer import _stub_forecast

    m2 = _stub_forecast("2017-09-06T12:00:00Z")
    result = kp_to_gic_risk(m2, bz=-28.0, speed=770.0, density=15.0)

    print(f"\nAlert: {result['headline_alert']['alert_level']}")
    print(f"Message: {result['headline_alert']['message']}")
    print(f"\nHorizons:")
    for h in result["horizons"]:
        print(
            f"  +{h['horizon_hr']:2d}h: Kp={h['kp_predicted']:.1f} "
            f"GIC={h['gic_mean']:.1f} [{h['gic_p5']:.1f}–{h['gic_p95']:.1f}] "
            f"Risk={h['risk_tier']}"
        )
