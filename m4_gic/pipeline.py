"""
ICARUS-X — M4 GIC: Full Pipeline

kp_to_gic_risk() — the main entry point called by model_runner.
Converts M2 Kp forecasts → GIC estimates → risk tiers → alerts.
Merges M3 classification results into the output.

Inputs:  M2 forecast dict + M3 classification dict
Outputs: M4 output contract dict (the final payload pushed to frontend)
"""

import sys
from pathlib import Path
from typing import Dict, Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from m4_gic.gic_model import kp_to_gic, gic_to_risk_tier
from m4_gic.uncertainty import estimate_gic_uncertainty
from m4_gic.alert_logic import determine_alert_level


def kp_to_gic_risk(
    m2_output: Dict[str, Any],
    m3_output: Dict[str, Any],
    bz: float = 0.0,
    speed: float = 400.0,
    density: float = 5.0,
) -> Dict[str, Any]:
    """
    Full M4 pipeline: Kp forecast → GIC risk assessment.

    Args:
        m2_output: M2 forecast dict with 'horizons' list
        m3_output: M3 classification dict with G-tier probs
        bz: Current Bz for uncertainty estimation
        speed: Current solar wind speed
        density: Current proton density

    Returns:
        M4 output contract dict
    """
    try:
        horizons = []

        for h in m2_output.get("horizons", []):
            kp = h.get("kp_predicted", 0)
            kp_low = h.get("kp_ci_low", kp - 1)
            kp_high = h.get("kp_ci_high", kp + 1)

            # GIC estimate with uncertainty
            gic_result = estimate_gic_uncertainty(kp, bz, speed, density)

            # Risk tier from mean GIC
            risk = gic_to_risk_tier(gic_result["gic_mean"])

            # Worst-case risk from p95 GIC
            risk_worst = gic_to_risk_tier(gic_result["gic_p95"])

            horizons.append({
                "horizon_hr": h.get("horizon_hr", 0),
                "kp_predicted": round(kp, 2),
                "kp_ci_low": round(kp_low, 2),
                "kp_ci_high": round(kp_high, 2),
                "gic_mean": gic_result["gic_mean"],
                "gic_p5": gic_result["gic_p5"],
                "gic_p95": gic_result["gic_p95"],
                "risk_tier": risk["tier"],
                "risk_tier_worst": risk_worst["tier"],
                "risk_color": risk["color"],
            })

        # Headline alert
        headline_alert = determine_alert_level(horizons)

        # Merge M3 G-tier probabilities
        g_tier_probs = {
            "g0": m3_output.get("g0_prob", 0),
            "g1": m3_output.get("g1_prob", 0),
            "g2": m3_output.get("g2_prob", 0),
            "g3": m3_output.get("g3_prob", 0),
            "g4": m3_output.get("g4_prob", 0),
            "predicted_tier": m3_output.get("predicted_tier", "G0"),
        }

        result = {
            "run_timestamp": m2_output.get("run_timestamp", ""),
            "horizons": horizons,
            "headline_alert": headline_alert,
            "g_tier_probs": g_tier_probs,
            "is_stub": m2_output.get("is_stub", False) or m3_output.get("is_stub", False),
        }

        logger.info(
            f"[M4] M4 Pipeline: Alert={headline_alert['alert_level']}, "
            f"Peak GIC={headline_alert['peak_gic_estimate']:.1f} A/km @ "
            f"+{headline_alert['peak_horizon_hr']}h"
        )

        return result

    except Exception as e:
        logger.error(f"[ERR] M4 pipeline error: {e}")
        return _stub_gic_risk()


def _stub_gic_risk() -> Dict[str, Any]:
    """Return a safe default M4 output on error."""
    return {
        "run_timestamp": "",
        "horizons": [],
        "headline_alert": {
            "alert_level": "LOW",
            "alert_color": "#4CAF50",
            "peak_horizon_hr": 0,
            "peak_gic_estimate": 0.0,
            "message": "System initializing — no forecast available.",
            "should_email": False,
            "should_sms": False,
        },
        "g_tier_probs": {
            "g0": 1.0, "g1": 0, "g2": 0, "g3": 0, "g4": 0,
            "predicted_tier": "G0",
        },
        "is_stub": True,
    }


if __name__ == "__main__":
    # Quick test with mock data
    from m2_predictor.infer import _stub_forecast
    from m3_classifier.infer import _stub_classify

    m2 = _stub_forecast("2017-09-06T12:00:00")
    m3 = _stub_classify()
    result = kp_to_gic_risk(m2, m3, bz=-20.0, speed=700.0, density=15.0)

    print(f"\nAlert: {result['headline_alert']['alert_level']}")
    print(f"Message: {result['headline_alert']['message']}")
    print(f"\nHorizons:")
    for h in result["horizons"]:
        print(
            f"  +{h['horizon_hr']:2d}h: Kp={h['kp_predicted']:.1f} "
            f"GIC={h['gic_mean']:.1f} [{h['gic_p5']:.1f}–{h['gic_p95']:.1f}] "
            f"Risk={h['risk_tier']}"
        )
