"""
ICARUS-X — M3 Classifier: XGBoost Training with Focal Loss + SMOTE

Trains XGBoost multiclass classifier for G0–G4 storm tier prediction.
Uses SMOTE to handle class imbalance and custom focal loss for
focusing on rare severe storms.

Inputs:  Feature matrix from features.py + G-tier labels
Outputs: Trained XGBoost model at models/xgb_sentinel.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from loguru import logger

from m3_classifier.storm_events import generate_synthetic_storm_events, kp_to_g_tier
from m3_classifier.features import build_feature_matrix, get_feature_names
from m2_predictor.data_loader import load_omni_csv, load_kp_csv, merge_datasets

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
XGB_PATH = MODELS_DIR / "xgb_sentinel.json"


def focal_loss_objective(y_pred: np.ndarray, dtrain: xgb.DMatrix, gamma: float = 2.0):
    """Custom focal loss objective for XGBoost to focus on hard examples."""
    labels = dtrain.get_label().astype(int)
    n_classes = 5

    # Softmax probabilities
    preds = y_pred.reshape(-1, n_classes)
    preds = np.exp(preds) / np.exp(preds).sum(axis=1, keepdims=True)

    # One-hot encode labels
    one_hot = np.zeros_like(preds)
    one_hot[np.arange(len(labels)), labels] = 1

    # Focal loss gradient and hessian
    pt = np.sum(preds * one_hot, axis=1, keepdims=True)
    pt = np.clip(pt, 1e-7, 1 - 1e-7)

    grad = preds - one_hot
    focal_weight = (1 - pt) ** gamma
    grad = (focal_weight * grad).reshape(-1)

    hess = preds * (1 - preds)
    hess = (focal_weight * hess).reshape(-1)
    hess = np.maximum(hess, 1e-7)

    return grad, hess


def train_classifier() -> None:
    """Train XGBoost storm classifier."""
    logger.info("[TRAIN] Training M3 Sentinel (XGBoost G-tier classifier)")

    # ── Build dataset ────────────────────────────────────
    sw_df = load_omni_csv()
    kp_df = load_kp_csv()
    merged = merge_datasets(sw_df, kp_df)

    # Build features
    feature_df = build_feature_matrix(merged)
    feature_df["g_tier"] = feature_df["current_kp_value"].apply(kp_to_g_tier)

    feature_names = [c for c in feature_df.columns if c not in ["timestamp", "g_tier"]]
    X = feature_df[feature_names].fillna(0).values
    y = feature_df["g_tier"].values

    logger.info(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # ── Train/test split ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── SMOTE oversampling ───────────────────────────────
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y_train)) - 1))
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info(f"   SMOTE: {len(X_train)} → {len(X_train_res)} samples")
    except Exception as e:
        logger.warning(f"[!] SMOTE failed ({e}), using original data")
        X_train_res, y_train_res = X_train, y_train

    # ── Train XGBoost ────────────────────────────────────
    dtrain = xgb.DMatrix(X_train_res, label=y_train_res, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    params = {
        "max_depth": 6,
        "eta": 0.1,
        "num_class": 5,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "seed": 42,
        "verbosity": 1,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dtest, "val")],
        early_stopping_rounds=20,
        verbose_eval=20,
    )

    # ── Evaluate ─────────────────────────────────────────
    y_pred = model.predict(dtest).astype(int)
    report = classification_report(y_test, y_pred, target_names=[f"G{i}" for i in range(5)])
    logger.info(f"\n[REPORT] Classification Report:\n{report}")

    # ── Save ─────────────────────────────────────────────
    model.save_model(str(XGB_PATH))
    logger.info(f"[SAVE] XGBoost model saved to {XGB_PATH}")

    # Save feature names for inference
    import json
    meta_path = MODELS_DIR / "xgb_sentinel_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"feature_names": feature_names}, f)
    logger.info(f"   Feature metadata saved to {meta_path}")


if __name__ == "__main__":
    train_classifier()
