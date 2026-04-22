"""
ICARUS-X — M5 Architect: Model Runner (Orchestrator)

Orchestrates the full M1 → M2 → M3 → M4 pipeline.
Automatically detects which models are available and uses
stubs for any missing components.

Called every 60 seconds by the WebSocket push loop in main.py.

Inputs:  Latest solar wind readings from poller
Outputs: Complete M4 output dict (pushed to frontend via WebSocket)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_pipeline(
    solar_wind_readings: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Execute the full forecasting pipeline.

    Args:
        solar_wind_readings: Recent solar wind data from NOAA poller

    Returns:
        M4 output contract dict
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.info(f"🔄 Pipeline run at {timestamp}")

    try:
        # ── M1: AR Features ──────────────────────────────
        ar_features = _run_m1(timestamp)

        # ── M2: Kp Forecast ──────────────────────────────
        input_window = _build_input_window(solar_wind_readings, ar_features)
        m2_output = _run_m2(input_window, timestamp)

        # ── M3: Storm Classification ─────────────────────
        sw_features = _build_classification_features(solar_wind_readings)
        m3_output = _run_m3(sw_features)

        # ── M4: GIC Risk Assessment ──────────────────────
        bz, speed, density = _get_current_conditions(solar_wind_readings)
        m4_output = _run_m4(m2_output, m3_output, bz, speed, density)

        logger.info(
            f"✅ Pipeline complete: Alert={m4_output['headline_alert']['alert_level']}, "
            f"Stub={m4_output.get('is_stub', True)}"
        )
        return m4_output

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return _fallback_output(timestamp)


def _run_m1(timestamp: str) -> Dict[str, Any]:
    """Run M1 Visionary (real or stub)."""
    try:
        use_real = os.environ.get("USE_REAL_M1", "false").lower() == "true"
        yolo_path = Path(os.environ.get("YOLO_CHECKPOINT", "models/yolov10/best.pt"))

        if use_real and yolo_path.exists():
            from m1_visionary.visionary import extract_features
            return extract_features()
        else:
            from m1_visionary.visionary_stub import get_ar_feature_vector
            return get_ar_feature_vector(timestamp)
    except Exception as e:
        logger.error(f"❌ M1 error: {e}")
        from m1_visionary.visionary_stub import get_ar_feature_vector
        return get_ar_feature_vector(timestamp)


def _run_m2(input_window: np.ndarray, timestamp: str) -> Dict[str, Any]:
    """Run M2 Predictor (real or stub)."""
    try:
        from m2_predictor.infer import run_forecast
        return run_forecast(input_window, timestamp)
    except Exception as e:
        logger.error(f"❌ M2 error: {e}")
        from m2_predictor.infer import _stub_forecast
        return _stub_forecast(timestamp)


def _run_m3(features: Dict[str, float]) -> Dict[str, Any]:
    """Run M3 Classifier (real or stub)."""
    try:
        from m3_classifier.infer import classify_window
        return classify_window(features)
    except Exception as e:
        logger.error(f"❌ M3 error: {e}")
        from m3_classifier.infer import _stub_classify
        return _stub_classify()


def _run_m4(
    m2_output: Dict, m3_output: Dict,
    bz: float, speed: float, density: float,
) -> Dict[str, Any]:
    """Run M4 GIC Pipeline."""
    try:
        from m4_gic.pipeline import kp_to_gic_risk
        return kp_to_gic_risk(m2_output, m3_output, bz, speed, density)
    except Exception as e:
        logger.error(f"❌ M4 error: {e}")
        from m4_gic.pipeline import _stub_gic_risk
        return _stub_gic_risk()


def _build_input_window(
    readings: Optional[List[Dict]],
    ar_features: Dict[str, Any],
) -> np.ndarray:
    """Build 24-hour input window for M2 from solar wind + AR features."""
    window_size = 24
    n_sw_features = 7  # bz, by, bx, bt, speed, density, temperature
    n_ar_features = 12
    n_total = n_sw_features + n_ar_features

    if readings and len(readings) >= window_size:
        recent = readings[-window_size:]
        sw_cols = ["bx_gsm", "by_gsm", "bz_gsm", "bt", "speed", "density", "temperature"]
        sw_data = []
        for r in recent:
            row = [float(r.get(c, 0) or 0) for c in sw_cols]
            sw_data.append(row)
        sw_array = np.array(sw_data, dtype=np.float32)
    else:
        # Generate synthetic current-like data
        rng = np.random.RandomState()
        sw_array = np.column_stack([
            rng.normal(0, 3, window_size),     # bx
            rng.normal(0, 4, window_size),     # by
            rng.normal(-5, 8, window_size),    # bz
            np.abs(rng.normal(8, 4, window_size)),  # bt
            rng.normal(450, 80, window_size),  # speed
            rng.lognormal(1.5, 0.5, window_size),  # density
            rng.lognormal(11, 0.3, window_size),    # temperature
        ]).astype(np.float32)

    # AR features (repeat for all timesteps)
    ar_vec = np.array([ar_features.get(f"f{i}", 0.0) for i in range(12)], dtype=np.float32)
    ar_tiled = np.tile(ar_vec, (window_size, 1))

    return np.hstack([sw_array, ar_tiled])


def _build_classification_features(
    readings: Optional[List[Dict]],
) -> Dict[str, float]:
    """Build feature dict for M3 from recent readings."""
    try:
        from m3_classifier.features import build_realtime_features
        return build_realtime_features(readings or [])
    except Exception:
        return {}


def _get_current_conditions(
    readings: Optional[List[Dict]],
) -> tuple:
    """Extract current Bz, speed, density from latest reading."""
    if readings and len(readings) > 0:
        latest = readings[-1]
        bz = float(latest.get("bz_gsm", 0) or 0)
        speed = float(latest.get("speed", 400) or 400)
        density = float(latest.get("density", 5) or 5)
        return bz, speed, density
    return 0.0, 400.0, 5.0


def _fallback_output(timestamp: str) -> Dict[str, Any]:
    """Return safe fallback when entire pipeline fails."""
    from m4_gic.pipeline import _stub_gic_risk
    result = _stub_gic_risk()
    result["run_timestamp"] = timestamp
    return result
