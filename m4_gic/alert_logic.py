"""
ICARUS-X — M4 GIC: Alert Logic

Determines alert levels and generates human-readable alert messages
based on GIC risk tiers across all forecast horizons.

Inputs:  List of per-horizon GIC risk assessments
Outputs: Headline alert dict with level, message, and notification flags
"""

from typing import Dict, Any, List
from loguru import logger

# ── Alert level hierarchy ────────────────────────────────
ALERT_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
ALERT_COLORS = {
    "LOW": "#4CAF50",
    "MEDIUM": "#FFA500",
    "HIGH": "#FF5722",
    "CRITICAL": "#D50000",
}

# ── Notification thresholds ──────────────────────────────
EMAIL_THRESHOLD = "HIGH"
SMS_THRESHOLD = "CRITICAL"


def determine_alert_level(horizons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Determine the headline alert from all forecast horizons.

    Args:
        horizons: List of per-horizon dicts with risk_tier, gic_mean, etc.

    Returns:
        Headline alert dict matching M4 output contract
    """
    if not horizons:
        return _default_alert()

    # Find the worst-case horizon
    worst_level_idx = 0
    worst_horizon = horizons[0]

    for h in horizons:
        tier = h.get("risk_tier", "LOW")
        idx = ALERT_LEVELS.index(tier) if tier in ALERT_LEVELS else 0
        if idx > worst_level_idx:
            worst_level_idx = idx
            worst_horizon = h

    alert_level = ALERT_LEVELS[worst_level_idx]

    # Build message
    peak_hr = worst_horizon.get("horizon_hr", 0)
    peak_gic = worst_horizon.get("gic_mean", 0)

    messages = {
        "LOW": f"Space weather quiet. No significant GIC risk expected.",
        "MEDIUM": f"Moderate geomagnetic activity expected in {peak_hr} hrs. GIC risk MEDIUM.",
        "HIGH": f"Geomagnetic storm expected in {peak_hr} hrs. GIC risk HIGH. Peak estimate: {peak_gic:.1f} A/km.",
        "CRITICAL": f"[CRITICAL] SEVERE geomagnetic storm in {peak_hr} hrs! GIC risk CRITICAL. Peak: {peak_gic:.1f} A/km. Immediate action required.",
    }

    should_email = worst_level_idx >= ALERT_LEVELS.index(EMAIL_THRESHOLD)
    should_sms = worst_level_idx >= ALERT_LEVELS.index(SMS_THRESHOLD)

    return {
        "alert_level": alert_level,
        "alert_color": ALERT_COLORS[alert_level],
        "peak_horizon_hr": peak_hr,
        "peak_gic_estimate": round(peak_gic, 2),
        "message": messages[alert_level],
        "should_email": should_email,
        "should_sms": should_sms,
    }


def _default_alert() -> Dict[str, Any]:
    """Return a safe default alert when no data is available."""
    return {
        "alert_level": "LOW",
        "alert_color": ALERT_COLORS["LOW"],
        "peak_horizon_hr": 0,
        "peak_gic_estimate": 0.0,
        "message": "No forecast data available. System initializing.",
        "should_email": False,
        "should_sms": False,
    }
