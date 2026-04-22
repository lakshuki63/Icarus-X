"""
ICARUS-X — M2 Predictor: Inference + Stub

run_forecast() — the main entry point called by model_runner.
Returns 8-horizon Kp forecast with Bayesian confidence intervals.
Falls back to stub if model checkpoint doesn't exist.

Inputs:  24-hour window of solar wind + AR features (19-dim)
Outputs: M2 output contract dict (see README)
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import joblib
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from m2_predictor.model import BiGRUPredictor
from m2_predictor.windowing import FORECAST_HORIZONS

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "bigru_predictor.pt"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"

# ── Module-level model cache ────────────────────────────
_model: Optional[BiGRUPredictor] = None
_device: str = "cpu"


def _load_model() -> Optional[BiGRUPredictor]:
    """Load trained BiGRU model from checkpoint."""
    global _model, _device

    if _model is not None:
        return _model

    if not CHECKPOINT_PATH.exists():
        logger.warning("[!] STUB MODE -- BiGRU checkpoint not found")
        return None

    try:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(CHECKPOINT_PATH, map_location=_device, weights_only=False)
        n_features = ckpt.get("n_features", 19)

        _model = BiGRUPredictor(input_size=n_features).to(_device)
        _model.load_state_dict(ckpt["model_state_dict"])
        _model.eval()
        logger.info(f"[OK] BiGRU model loaded (epoch {ckpt.get('epoch', '?')})")
        return _model
    except Exception as e:
        logger.error(f"[ERR] Failed to load BiGRU model: {e}")
        return None


def run_forecast(
    input_window: np.ndarray,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Kp forecast on a 24-hour input window.

    Args:
        input_window: (24, n_features) numpy array, already normalized
        timestamp: ISO timestamp for this forecast run

    Returns:
        M2 output contract dict
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    model = _load_model()

    if model is not None:
        return _real_forecast(model, input_window, timestamp)
    else:
        return _stub_forecast(timestamp)


def _real_forecast(
    model: BiGRUPredictor,
    input_window: np.ndarray,
    timestamp: str,
) -> Dict[str, Any]:
    """Run real model inference with MC Dropout uncertainty."""
    try:
        x = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0).to(_device)

        result = model.predict_with_uncertainty(x, n_samples=30)

        horizons = []
        for i, h in enumerate(FORECAST_HORIZONS):
            attn = result["attention_weights"][0].cpu().numpy().tolist()
            # Pad/truncate attention to 1440 for contract compliance
            if len(attn) < 1440:
                attn = attn + [0.0] * (1440 - len(attn))
            else:
                attn = attn[:1440]

            horizons.append({
                "horizon_hr": h,
                "kp_predicted": round(float(result["kp_pred"][0, i].cpu()), 2),
                "kp_ci_low": round(float(result["kp_ci_low"][0, i].cpu()), 2),
                "kp_ci_high": round(float(result["kp_ci_high"][0, i].cpu()), 2),
                "kp_std": round(float(result["kp_std"][0, i].cpu()), 2),
                "attention_weights": attn,
            })

        return {
            "run_timestamp": timestamp,
            "horizons": horizons,
            "is_stub": False,
        }
    except Exception as e:
        logger.error(f"[ERR] Real forecast failed, falling back to stub: {e}")
        return _stub_forecast(timestamp)


def _stub_forecast(timestamp: str) -> Dict[str, Any]:
    """Generate realistic stub forecast based on Sept 2017 storm profile."""
    logger.warning("[!] STUB MODE -- using mock Kp forecast")
    rng = np.random.RandomState()

    # Realistic storm-like profile: Kp rises then falls
    base_kp = [5.2, 6.1, 7.3, 7.8, 6.9, 5.8, 4.7, 3.9]
    noise = rng.normal(0, 0.3, 8)

    # Generate smooth attention weights (24 timesteps, padded to 1440)
    raw_attn = np.abs(rng.normal(0, 1, 24))
    raw_attn = raw_attn / raw_attn.sum()
    attn_padded = np.zeros(1440)
    for i, w in enumerate(raw_attn):
        attn_padded[i * 60:(i + 1) * 60] = w / 60

    horizons = []
    for i, h in enumerate(FORECAST_HORIZONS):
        kp = float(np.clip(base_kp[i] + noise[i], 0, 9))
        std = float(rng.uniform(0.4, 1.0))
        horizons.append({
            "horizon_hr": h,
            "kp_predicted": round(kp, 2),
            "kp_ci_low": round(max(0, kp - 1.5 * std), 2),
            "kp_ci_high": round(min(9, kp + 1.5 * std), 2),
            "kp_std": round(std, 2),
            "attention_weights": attn_padded.tolist(),
        })

    return {
        "run_timestamp": timestamp,
        "horizons": horizons,
        "is_stub": True,
    }


if __name__ == "__main__":
    result = run_forecast(np.random.randn(24, 19))
    print(f"Forecast at: {result['run_timestamp']}")
    print(f"Stub mode:   {result.get('is_stub', False)}")
    for h in result["horizons"]:
        print(f"  +{h['horizon_hr']:2d}h: Kp={h['kp_predicted']:.1f} [{h['kp_ci_low']:.1f}–{h['kp_ci_high']:.1f}]")
