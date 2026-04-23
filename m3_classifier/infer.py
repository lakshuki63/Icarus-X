"""
ICARUS-X — M3 Sentinel: Solar Flare Classifier Inference (INDEPENDENT MODULE)

classify_flare() — called by M5 model_runner independently from M2/M4.
Predicts the probability of an M-class or X-class solar flare from
6 SHARP photospheric parameters.

This module has NO dependency on M1, M2, or M4.

Inputs:
  feature_dict: Dict with keys f0–f5 (SHARP values OR M1 proxy values)
  Values are in physical units — log1p transform applied internally.

Outputs (M3 contract):
  {
    "flare_probability": float [0,1],
    "flare_class":       "none" | "C" | "M" | "X",
    "predicted_tier":    "Quiet" | "C-class" | "M-class" | "X-class",
    "confidence":        "LOW" | "MEDIUM" | "HIGH",
    "top_features": [
      {"name": "total_unsigned_current_helicity (TOTUSJH)", "shap_value": 0.51},
      ...
    ],
    "is_stub": bool
  }
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

# ── Project root — ZERO M2 IMPORTS ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from m3_classifier.features import (
    FEATURE_COLS,
    FEATURE_DESCRIPTIONS,
    prepare_inference_features,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = PROJECT_ROOT / "models"
MODEL_PATH  = MODELS_DIR / "xgb_flare_sentinel.json"
META_PATH   = MODELS_DIR / "xgb_flare_sentinel_meta.json"

# ── Module-level cache ────────────────────────────────────────────────────────
_model = None
_meta: Optional[Dict] = None


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model():
    """
    Load XGBoost flare model and metadata from checkpoint.

    Raises:
        FileNotFoundError: If checkpoint does not exist (fail loudly — no silent stub)
    """
    global _model, _meta

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"M3 model checkpoint not found at: {MODEL_PATH}\n"
            "Run training first:\n"
            "  python m3_classifier/data_download.py --source donki\n"
            "  python m3_classifier/train_xgb.py"
        )

    try:
        import xgboost as xgb
        _model = xgb.Booster()
        _model.load_model(str(MODEL_PATH))
        logger.info(f"[M3] XGBoost flare model loaded from {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load M3 model from {MODEL_PATH}: {e}\n"
            "The checkpoint may be corrupt. Re-run:\n"
            "  python m3_classifier/train_xgb.py"
        )

    if META_PATH.exists():
        with open(META_PATH) as f:
            _meta = json.load(f)
        logger.info(
            f"[M3] Metadata loaded: "
            f"F2={_meta.get('test_f2','?')}, "
            f"TSS={_meta.get('test_tss','?')}, "
            f"threshold={_meta.get('optimal_threshold', 0.5)}"
        )
    else:
        logger.warning(f"[M3] Metadata not found at {META_PATH} — using defaults")
        _meta = {"optimal_threshold": 0.5, "feature_names": FEATURE_COLS}

    return _model


def _get_threshold() -> float:
    """Return the optimal probability threshold from training metadata."""
    if _meta is not None:
        return float(_meta.get("optimal_threshold", 0.5))
    return 0.5


# ── Flare class mapping ───────────────────────────────────────────────────────
def _probability_to_class(prob: float, threshold: float) -> Dict[str, str]:
    """
    Map flare probability to class label and tier.

    Binary model outputs P(M/X flare). We further distinguish M vs X
    by probability magnitude above threshold (heuristic).

    Args:
        prob: Flare probability in [0, 1]
        threshold: Optimal threshold from training

    Returns:
        Dict with flare_class, predicted_tier, confidence
    """
    if prob < threshold:
        return {
            "flare_class":    "none",
            "predicted_tier": "Quiet",
            "confidence":     "HIGH" if prob < threshold * 0.3 else "MEDIUM",
        }

    # Above threshold → M or X class
    # X-class heuristic: probability ≥ 0.85 OR ≥ 2× threshold
    x_threshold = min(0.85, threshold * 2.0)
    if prob >= x_threshold:
        return {
            "flare_class":    "X",
            "predicted_tier": "X-class",
            "confidence":     "HIGH" if prob >= 0.90 else "MEDIUM",
        }
    else:
        return {
            "flare_class":    "M",
            "predicted_tier": "M-class",
            "confidence":     "HIGH" if prob >= threshold * 1.5 else "MEDIUM",
        }


# ── SHAP per-sample ───────────────────────────────────────────────────────────
def _compute_shap_per_sample(
    model,
    x: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Compute SHAP values for a single inference sample.

    Args:
        model: Loaded XGBoost Booster
        x: (1, 6) feature array

    Returns:
        Top-3 features sorted by absolute SHAP value
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(x)[0]  # shape (6,)

        results = [
            {
                "name": FEATURE_DESCRIPTIONS.get(feat, feat),
                "shap_value": round(float(sv), 4),
            }
            for feat, sv in zip(FEATURE_COLS, shap_vals)
        ]
        results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return results[:3]

    except Exception:
        # Fallback: use model gain importance (not sample-specific but always works)
        try:
            scores = model.get_score(importance_type="gain")
            total = sum(scores.values()) or 1.0
            results = [
                {
                    "name": FEATURE_DESCRIPTIONS.get(k, k),
                    "shap_value": round(v / total, 4),
                }
                for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ]
            return results[:3]
        except Exception:
            return [
                {"name": FEATURE_DESCRIPTIONS[feat], "shap_value": 0.0}
                for feat in FEATURE_COLS[:3]
            ]


# ── Main inference entry point ────────────────────────────────────────────────
def classify_flare(
    feature_dict: Dict[str, float],
) -> Dict[str, Any]:
    """
    Classify solar flare probability from SHARP feature dict.

    Called by M5 model_runner independently — does NOT depend on M2 or M4.

    Args:
        feature_dict: Dict with keys f0–f5 in physical units.
                      Can be raw SHARP values OR M1 proxy values
                      (via features.map_m1_to_m3_features).

    Returns:
        M3 output contract dict:
        {
            "flare_probability": float,   # [0.0, 1.0]
            "flare_class":       str,     # "none" | "C" | "M" | "X"
            "predicted_tier":    str,     # "Quiet" | "C-class" | "M-class" | "X-class"
            "confidence":        str,     # "LOW" | "MEDIUM" | "HIGH"
            "top_features":      list,    # [{name, shap_value}, ...]
            "is_stub":           bool
        }

    Raises:
        FileNotFoundError: If model checkpoint is missing (caught by model_runner)
    """
    import xgboost as xgb

    model = _load_model()  # Raises FileNotFoundError if missing — correct behavior

    # Apply same log1p transform used during training
    x = prepare_inference_features(feature_dict)  # (1, 6) float32

    # Predict probability
    dmat = xgb.DMatrix(x, feature_names=FEATURE_COLS)
    raw_prob = float(model.predict(dmat)[0])

    # Clip and assert
    prob = float(np.clip(raw_prob, 0.0, 1.0))
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

    # Map to class
    threshold = _get_threshold()
    class_info = _probability_to_class(prob, threshold)

    # SHAP top-3
    top_features = _compute_shap_per_sample(model, x)

    result = {
        "flare_probability": round(prob, 4),
        "flare_class":       class_info["flare_class"],
        "predicted_tier":    class_info["predicted_tier"],
        "confidence":        class_info["confidence"],
        "top_features":      top_features,
        "is_stub":           False,
    }

    logger.info(
        f"[M3] Flare: prob={prob:.3f}, class={class_info['flare_class']}, "
        f"tier={class_info['predicted_tier']}, confidence={class_info['confidence']}"
    )
    return result


# ── Validation helpers ────────────────────────────────────────────────────────
def validate_output(result: Dict[str, Any]) -> None:
    """
    Assert M3 output contract compliance.

    Args:
        result: Output dict from classify_flare()

    Raises:
        AssertionError: If contract is violated
    """
    prob = result.get("flare_probability", -1)
    assert 0.0 <= prob <= 1.0, f"flare_probability out of range: {prob}"

    valid_classes = {"none", "C", "M", "X"}
    assert result.get("flare_class") in valid_classes, \
        f"Invalid flare_class: {result.get('flare_class')}"

    valid_tiers = {"Quiet", "C-class", "M-class", "X-class"}
    assert result.get("predicted_tier") in valid_tiers, \
        f"Invalid predicted_tier: {result.get('predicted_tier')}"

    valid_confidence = {"LOW", "MEDIUM", "HIGH"}
    assert result.get("confidence") in valid_confidence, \
        f"Invalid confidence: {result.get('confidence')}"

    top = result.get("top_features", [])
    assert isinstance(top, list) and len(top) > 0, "top_features must be non-empty list"


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run M3 flare inference")
    parser.add_argument(
        "--scenario", choices=["quiet", "active", "extreme"],
        default="active",
        help="Test scenario (quiet/active/extreme)"
    )
    args = parser.parse_args()

    # Test feature sets (approximate SHARP values for each scenario)
    SCENARIOS = {
        "quiet": {
            "f0": 1e20, "f1": 5e23, "f2": 1e11, "f3": 5e13, "f4": 50.0,  "f5": 30.0,
        },
        "active": {
            "f0": 5e22, "f1": 2e25, "f2": 5e13, "f3": 2e15, "f4": 2000.0, "f5": 500.0,
        },
        "extreme": {    # Sept 2017 X9.3-like
            "f0": 2e24, "f1": 8e26, "f2": 3e14, "f3": 1e16, "f4": 8000.0, "f5": 2000.0,
        },
    }

    features = SCENARIOS[args.scenario]
    print(f"\nScenario: {args.scenario.upper()}")
    print(f"Input features:")
    for k, v in features.items():
        print(f"  {k}: {v:.2e}")

    try:
        result = classify_flare(features)
        validate_output(result)

        print(f"\nM3 Output:")
        print(f"  flare_probability: {result['flare_probability']}")
        print(f"  flare_class:       {result['flare_class']}")
        print(f"  predicted_tier:    {result['predicted_tier']}")
        print(f"  confidence:        {result['confidence']}")
        print(f"  is_stub:           {result['is_stub']}")
        print(f"  top_features:")
        for f in result["top_features"]:
            print(f"    {f['name']}: {f['shap_value']}")

        print("\n[PASS] Contract validation passed.")

    except FileNotFoundError as e:
        print(f"\n[INFO] Model not trained yet:\n{e}")
        print("\nExpected output format:")
        print({
            "flare_probability": 0.83,
            "flare_class": "X",
            "predicted_tier": "X-class",
            "confidence": "HIGH",
            "top_features": [
                {"name": "free_energy_proxy (TOTPOT)", "shap_value": 0.51},
                {"name": "total_unsigned_current_helicity (TOTUSJH)", "shap_value": 0.33},
                {"name": "pil_flux (R_VALUE)", "shap_value": 0.21},
            ],
            "is_stub": False,
        })
