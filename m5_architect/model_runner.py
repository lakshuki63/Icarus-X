"""
ICARUS-X — M5 Architect: Model Runner (Orchestrator)

Correct pipeline (M3 is FULLY INDEPENDENT of M1/M2/M4):

  [Solar Wind DB] → M2 (Kp forecast) → M4 (GIC risk)
  [AR features]  → M1 proxy → M3 (flare prob) [independent]
  M5 merges all 4 outputs into one WebSocket payload.

Output contract (pushed every 60s via WebSocket):
  {
    "timestamp":         str,
    "kp_forecast":       M2 output dict,
    "flare":             M3 output dict,
    "gic_risk":          M4 output dict,
    "solar_wind_latest": dict,
    "data_quality":      {m1_real, m2_real, m3_real, m4_real}
  }
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "solar_wind_window": 1440,   # 24h at 1-min cadence
    "ar_features_path":  PROJECT_ROOT / "data" / "ar_features.csv",
    "yolo_checkpoint":   PROJECT_ROOT / "models" / "yolov10" / "best.pt",
    "ar_staleness_hrs":  12,     # AR features older than this → use zeros
}

HORIZONS = [3, 6, 9, 12, 15, 18, 21, 24]


# ── Step 1: Solar wind ────────────────────────────────────────────────────────
def _load_solar_wind(readings: Optional[List[Dict]]) -> List[Dict]:
    """Return up to 1440 most recent solar wind readings, padded with zeros if short."""
    if not readings:
        logger.warning("[Runner] No solar wind readings — using zero-padded window")
        return []
    window = readings[-CONFIG["solar_wind_window"]:]
    if len(window) < CONFIG["solar_wind_window"]:
        logger.warning(
            f"[Runner] Solar wind window short: {len(window)}/{CONFIG['solar_wind_window']} rows"
        )
    return window


def _solar_wind_to_dataframe(readings: List[Dict]):
    """Convert readings list to numpy array for M2."""
    import pandas as pd
    if not readings:
        return np.zeros((24, 7), dtype=np.float32)
    df = pd.DataFrame(readings)
    cols = ["bx_gsm", "by_gsm", "bz_gsm", "bt", "speed", "density", "temperature"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    arr = df[cols].fillna(0.0).values.astype(np.float32)
    # Downsample/pad to exactly 24 rows (hourly)
    if len(arr) >= 60:
        arr = arr[::60][:24]  # every 60th row (1-min → 1-hr)
    arr = arr[:24]
    if len(arr) < 24:
        pad = np.zeros((24 - len(arr), 7), dtype=np.float32)
        arr = np.vstack([pad, arr])
    return arr


# ── Step 2: AR features ───────────────────────────────────────────────────────
def _load_ar_features() -> tuple[Dict[str, float], bool]:
    """
    Load latest AR feature row from ar_features.csv.

    Returns:
        (ar_features_dict, is_real)
        is_real=False if file missing or all-zero or stale >12h
    """
    import pandas as pd
    from datetime import timedelta

    path = CONFIG["ar_features_path"]
    zero_vec = {f"f{i}": 0.0 for i in range(12)}

    if not path.exists():
        logger.warning(f"[Runner] AR features file not found: {path} — using zero vector")
        return zero_vec, False

    try:
        df = pd.read_csv(path)
        if df.empty:
            logger.warning("[Runner] ar_features.csv is empty — using zero vector")
            return zero_vec, False

        # Try to get most recent row by timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            if df.empty:
                return zero_vec, False

            latest_row = df.iloc[-1]
            age_hrs = (datetime.now(timezone.utc) - latest_row["timestamp"].to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds() / 3600
            if age_hrs > CONFIG["ar_staleness_hrs"]:
                logger.warning(f"[Runner] AR features stale ({age_hrs:.1f}h old) — using zero vector")
                return zero_vec, False
        else:
            latest_row = df.iloc[-1]

        ar = {f"f{i}": float(latest_row.get(f"f{i}", 0.0)) for i in range(12)}

        # Quality check: if f0 (total flux) is zero but regions were detected, warn
        if ar.get("f0", 0.0) == 0.0:
            logger.warning("[Runner] AR feature f0=0 — M1 may not be producing valid features")
            return ar, False

        zero_frac = sum(1 for v in ar.values() if v == 0.0) / 12
        if zero_frac > 0.5:
            logger.warning(f"[Runner] {zero_frac*100:.0f}% of AR features are zero — M1 output suspect")

        logger.info(f"[Runner] AR features loaded: f0={ar['f0']:.4f} (real M1 output)")
        return ar, True

    except Exception as e:
        logger.error(f"[Runner] Failed to load AR features: {e}")
        return zero_vec, False


# ── Step 3: Run M2 ────────────────────────────────────────────────────────────
def _run_m2(solar_wind_arr: np.ndarray, ar_features: Dict, timestamp: str) -> tuple[Dict, bool]:
    """Run M2 Kp forecast. Returns (output, is_real)."""
    try:
        ar_vec = np.array([ar_features.get(f"f{i}", 0.0) for i in range(12)], dtype=np.float32)
        ar_tiled = np.tile(ar_vec, (24, 1))
        input_window = np.hstack([solar_wind_arr, ar_tiled])  # (24, 19)

        from m2_predictor.infer import run_forecast
        result = run_forecast(input_window, timestamp)
        is_real = not result.get("is_stub", True)
        return result, is_real

    except FileNotFoundError as e:
        logger.error(f"[Runner] M2 checkpoint missing: {e}")
        return _fallback_m2(timestamp), False
    except Exception as e:
        logger.error(f"[Runner] M2 error: {e}")
        return _fallback_m2(timestamp), False


def _fallback_m2(timestamp: str) -> Dict:
    """Minimal M2 fallback — zeros, not fake storm data."""
    return {
        "run_timestamp": timestamp,
        "horizons": [
            {
                "horizon_hr": h,
                "kp_predicted": 0.0,
                "kp_ci_low": 0.0,
                "kp_ci_high": 0.0,
                "kp_std": 0.0,
                "attention_weights": [1.0 / 24] * 24,
            }
            for h in HORIZONS
        ],
        "is_stub": True,
    }


# ── Step 4: Run M3 (INDEPENDENT) ─────────────────────────────────────────────
def _run_m3(ar_features: Dict) -> tuple[Dict, bool]:
    """
    Run M3 flare classifier — INDEPENDENTLY of M2 output.

    Uses M1 AR features as proxy for SHARP parameters via
    features.map_m1_to_m3_features(). Does NOT use solar wind data.
    """
    try:
        from m3_classifier.features import map_m1_to_m3_features
        from m3_classifier.infer import classify_flare

        m3_input = map_m1_to_m3_features(ar_features)
        result = classify_flare(m3_input)
        is_real = not result.get("is_stub", True)
        return result, is_real

    except FileNotFoundError as e:
        logger.error(f"[Runner] M3 checkpoint missing: {e}")
        return _fallback_m3(), False
    except Exception as e:
        logger.error(f"[Runner] M3 error: {e}")
        return _fallback_m3(), False


def _fallback_m3() -> Dict:
    """Minimal M3 fallback — no flare predicted (conservative)."""
    return {
        "flare_probability": 0.0,
        "flare_class":       "none",
        "predicted_tier":    "Quiet",
        "confidence":        "LOW",
        "top_features": [
            {"name": "free_energy_proxy (TOTPOT)",                        "shap_value": 0.0},
            {"name": "total_unsigned_current_helicity (TOTUSJH)",         "shap_value": 0.0},
            {"name": "pil_flux (R_VALUE)",                                "shap_value": 0.0},
        ],
        "is_stub": True,
    }


# ── Step 5: Run M4 ────────────────────────────────────────────────────────────
def _run_m4(m2_output: Dict, bz: float, speed: float, density: float) -> tuple[Dict, bool]:
    """Run M4 GIC risk from M2 Kp output ONLY (M3 not passed to M4)."""
    try:
        from m4_gic.pipeline import kp_to_gic_risk
        result = kp_to_gic_risk(m2_output, bz=bz, speed=speed, density=density)
        return result, True
    except TypeError:
        # Older pipeline.py still expects m3_output — call with dummy
        try:
            from m4_gic.pipeline import kp_to_gic_risk
            result = kp_to_gic_risk(m2_output, {}, bz=bz, speed=speed, density=density)
            return result, True
        except Exception as e:
            logger.error(f"[Runner] M4 error: {e}")
            return _fallback_m4(), False
    except Exception as e:
        logger.error(f"[Runner] M4 error: {e}")
        return _fallback_m4(), False


def _fallback_m4() -> Dict:
    """Minimal M4 fallback — LOW alert, zero GIC."""
    return {
        "run_timestamp": "",
        "horizons": [
            {
                "horizon_hr": h,
                "kp_predicted": 0.0,
                "kp_ci_low": 0.0,
                "kp_ci_high": 0.0,
                "gic_mean": 0.13,
                "gic_p5": 0.05,
                "gic_p95": 0.3,
                "risk_tier": "LOW",
                "risk_tier_worst": "LOW",
                "risk_color": "#4CAF50",
            }
            for h in HORIZONS
        ],
        "headline_alert": {
            "alert_level": "LOW",
            "alert_color": "#4CAF50",
            "peak_horizon_hr": 0,
            "peak_gic_estimate": 0.13,
            "message": "Model initializing — no forecast available.",
            "should_email": False,
            "should_sms": False,
        },
        "is_stub": True,
    }


# ── Step 6: Merge ─────────────────────────────────────────────────────────────
def _get_current_conditions(readings: List[Dict]) -> tuple[float, float, float]:
    """Extract latest Bz, speed, density from solar wind readings."""
    if readings:
        last = readings[-1]
        return (
            float(last.get("bz_gsm", 0) or 0),
            float(last.get("speed",  400) or 400),
            float(last.get("density", 5) or 5),
        )
    return 0.0, 400.0, 5.0


def _latest_sw_reading(readings: List[Dict]) -> Dict:
    """Extract latest solar wind snapshot for WebSocket payload."""
    if not readings:
        return {"bz": 0.0, "vsw": 400.0, "np": 5.0, "pdyn": 1.5, "timestamp": None}
    last = readings[-1]
    return {
        "bz":        float(last.get("bz_gsm",      0)   or 0),
        "vsw":       float(last.get("speed",        400) or 400),
        "np":        float(last.get("density",      5)   or 5),
        "pdyn":      float(last.get("dynamic_pressure", 0) or 0),
        "timestamp": last.get("timestamp"),
    }


# ── Main entry point ──────────────────────────────────────────────────────────
def run_full_pipeline(
    solar_wind_readings: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Execute the full ICARUS-X forecasting pipeline.

    Correct data flow:
      M1 AR features (proxy) → M3 flare classifier (independent)
      Solar wind + AR features → M2 Kp forecast → M4 GIC risk
      M5 merges all four outputs into one WebSocket payload.

    Args:
        solar_wind_readings: Recent 1-min solar wind readings from poller

    Returns:
        Merged output dict matching WebSocket contract
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.info(f"[Runner] Pipeline run at {timestamp}")

    # Step 1 — Solar wind
    readings = _load_solar_wind(solar_wind_readings)
    sw_arr = _solar_wind_to_dataframe(readings)
    bz, speed, density = _get_current_conditions(readings)

    # Step 2 — AR features (from M1 output CSV)
    ar_features, m1_real = _load_ar_features()

    # Step 3 — M2: Kp forecast (uses solar wind + AR features)
    m2_output, m2_real = _run_m2(sw_arr, ar_features, timestamp)

    # Step 4 — M3: Flare probability (INDEPENDENT — uses AR proxy only)
    m3_output, m3_real = _run_m3(ar_features)

    # Step 5 — M4: GIC risk (uses M2 Kp output only)
    m4_output, m4_real = _run_m4(m2_output, bz, speed, density)

    # Step 6 — Merge into WebSocket contract
    result = {
        "timestamp":    timestamp,
        "kp_forecast":  m2_output,
        "flare":        m3_output,
        "gic_risk":     m4_output,
        "solar_wind_latest": _latest_sw_reading(readings),
        "data_quality": {
            "m1_real": m1_real,
            "m2_real": m2_real,
            "m3_real": m3_real,
            "m4_real": m4_real,
        },
    }

    alert = m4_output.get("headline_alert", {}).get("alert_level", "UNKNOWN")
    logger.info(
        f"[Runner] ✅ Complete | Alert={alert} | "
        f"M1={'real' if m1_real else 'proxy'} "
        f"M2={'real' if m2_real else 'fallback'} "
        f"M3={'real' if m3_real else 'fallback'} "
        f"M4={'real' if m4_real else 'fallback'}"
    )
    return result


# ── Backward-compatible alias (called by main.py) ─────────────────────────────
def run_pipeline(
    solar_wind_readings: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Alias for run_full_pipeline() — preserves main.py call signature."""
    return run_full_pipeline(solar_wind_readings)
