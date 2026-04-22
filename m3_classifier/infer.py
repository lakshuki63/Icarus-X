"""
ICARUS-X — M3 Classifier: Inference + Stub

classify_window() — called by model_runner to get G-tier probabilities.
Falls back to stub if XGBoost model doesn't exist.

Inputs:  Feature dict from recent solar wind window
Outputs: M3 output contract dict with G-tier probs + SHAP top features
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
XGB_PATH = MODELS_DIR / "xgb_sentinel.json"
META_PATH = MODELS_DIR / "xgb_sentinel_meta.json"

_model = None
_feature_names: Optional[List[str]] = None


def _load_model():
    """Load XGBoost model and feature metadata."""
    global _model, _feature_names

    if _model is not None:
        return _model

    if not XGB_PATH.exists():
        logger.warning("[!] STUB MODE -- XGBoost checkpoint not found")
        return None

    try:
        import xgboost as xgb
        _model = xgb.Booster()
        _model.load_model(str(XGB_PATH))

        if META_PATH.exists():
            with open(META_PATH) as f:
                meta = json.load(f)
            _feature_names = meta.get("feature_names", [])

        logger.info("[OK] XGBoost model loaded")
        return _model
    except Exception as e:
        logger.error(f"[ERR] Failed to load XGBoost: {e}")
        return None


def classify_window(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Classify storm severity from solar wind features.

    Args:
        features: Dict of feature_name → value

    Returns:
        M3 output contract dict
    """
    model = _load_model()

    if model is not None:
        return _real_classify(model, features)
    else:
        return _stub_classify()


def _real_classify(model, features: Dict[str, float]) -> Dict[str, Any]:
    """Run real XGBoost classification with SHAP-like feature importance."""
    try:
        import xgboost as xgb

        if _feature_names:
            x = np.array([[features.get(f, 0.0) for f in _feature_names]])
            dmat = xgb.DMatrix(x, feature_names=_feature_names)
        else:
            x = np.array([list(features.values())])
            dmat = xgb.DMatrix(x)

        probs = model.predict(dmat)[0]
        if len(probs.shape) == 0:
            # Single class prediction
            predicted_tier = int(probs)
            probs = np.zeros(5)
            probs[predicted_tier] = 1.0
        else:
            predicted_tier = int(np.argmax(probs))

        # Get feature importance as proxy for SHAP
        importance = model.get_score(importance_type="gain")
        top_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "g0_prob": round(float(probs[0]), 4),
            "g1_prob": round(float(probs[1]), 4),
            "g2_prob": round(float(probs[2]), 4),
            "g3_prob": round(float(probs[3]), 4),
            "g4_prob": round(float(probs[4] if len(probs) > 4 else 0), 4),
            "predicted_tier": f"G{predicted_tier}",
            "top_features": [
                {"name": name, "shap_value": round(val / max(1, sum(v for _, v in top_feats)), 2)}
                for name, val in top_feats
            ],
            "is_stub": False,
        }
    except Exception as e:
        logger.error(f"[ERR] Real classify failed: {e}")
        return _stub_classify()


def _stub_classify() -> Dict[str, Any]:
    """Generate realistic stub G-tier classification."""
    logger.warning("[!] STUB MODE -- using mock storm classification")
    rng = np.random.RandomState()

    # Simulate a moderate-to-strong storm scenario
    raw = rng.dirichlet([2, 5, 4, 3, 1])
    predicted_tier = int(np.argmax(raw))

    return {
        "g0_prob": round(float(raw[0]), 4),
        "g1_prob": round(float(raw[1]), 4),
        "g2_prob": round(float(raw[2]), 4),
        "g3_prob": round(float(raw[3]), 4),
        "g4_prob": round(float(raw[4]), 4),
        "predicted_tier": f"G{predicted_tier}",
        "top_features": [
            {"name": "min_bz_gsm_6hr", "shap_value": round(rng.uniform(0.3, 0.5), 2)},
            {"name": "max_speed_12hr", "shap_value": round(rng.uniform(0.2, 0.35), 2)},
            {"name": "mean_kp_value_6hr", "shap_value": round(rng.uniform(0.1, 0.25), 2)},
        ],
        "is_stub": True,
    }


if __name__ == "__main__":
    result = classify_window({})
    print(f"Predicted tier: {result['predicted_tier']}")
    print(f"Stub mode: {result.get('is_stub')}")
    for k in ["g0_prob", "g1_prob", "g2_prob", "g3_prob", "g4_prob"]:
        print(f"  {k}: {result[k]}")
