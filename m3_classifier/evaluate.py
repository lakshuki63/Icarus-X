"""
ICARUS-X — M3 Sentinel: Evaluation Script

Evaluates the trained XGBoost flare classifier on the held-out test set.
Computes TSS, F2-score, ROC-AUC, and skill scores used in solar physics.

Target metrics (from spec):
  TSS   > 0.65 for M+X class
  F2    > 0.70

Usage:
  python m3_classifier/evaluate.py
  python m3_classifier/evaluate.py --threshold 0.35
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from m3_classifier.features import (
    FEATURE_COLS, build_feature_matrix, load_sharp_dataset,
)

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "xgb_flare_sentinel.json"
META_PATH  = MODELS_DIR / "xgb_flare_sentinel_meta.json"

CONFIG = {
    "train_ratio": 0.70,
    "val_ratio":   0.15,
    # test = remaining 0.15
}


# ── Skill scores ──────────────────────────────────────────────────────────────
def compute_tss(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    True Skill Statistic (Hanssen-Kuipers Discriminant).
    TSS = sensitivity + specificity - 1 = TP/(TP+FN) - FP/(FP+TN)
    Range: [-1, 1]. TSS > 0.5 is considered skilful for flare forecasting.
    """
    sens = tp / max(tp + fn, 1)  # sensitivity / recall
    spec = tn / max(tn + fp, 1)  # specificity
    return float(sens + spec - 1.0)


def compute_hss(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Heidke Skill Score — measures improvement over random chance.
    HSS = 2(TP*TN - FP*FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN))
    """
    n = tp + tn + fp + fn
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / max(n, 1)
    correct  = tp + tn
    return float(2 * (correct - expected) / max(n - expected, 1e-9))


def compute_fbeta(
    tp: int, fp: int, fn: int, beta: float = 2.0
) -> float:
    """
    F-beta score — F2 weights recall 2x more than precision.
    Critical for rare flare events where missing a real flare is costly.
    """
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    b2 = beta ** 2
    denom = b2 * precision + recall
    return float((1 + b2) * precision * recall / max(denom, 1e-9))


def confusion_components(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """Return (TP, TN, FP, FN) from binary arrays."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn


# ── Threshold sweep ───────────────────────────────────────────────────────────
def threshold_sweep(
    probs: np.ndarray,
    y_true: np.ndarray,
    beta: float = 2.0,
) -> pd.DataFrame:
    """
    Sweep probability thresholds from 0.1 to 0.9 and compute metrics at each.

    Returns DataFrame with columns: threshold, TSS, F2, HSS, precision, recall, fp_rate.
    """
    rows = []
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (probs >= t).astype(int)
        tp, tn, fp, fn = confusion_components(y_true, y_pred)

        tss  = compute_tss(tp, tn, fp, fn)
        f2   = compute_fbeta(tp, fp, fn, beta=beta)
        hss  = compute_hss(tp, tn, fp, fn)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        fpr  = fp / max(fp + tn, 1)

        rows.append({
            "threshold": round(float(t), 2),
            "TSS": round(tss, 4),
            "F2":  round(f2,  4),
            "HSS": round(hss, 4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "fp_rate":   round(fpr,  4),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        })

    return pd.DataFrame(rows)


# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate(threshold: float = None) -> Dict:
    """
    Full evaluation of M3 on the held-out test set.

    Args:
        threshold: Probability threshold (None = use training metadata value)

    Returns:
        Dict of evaluation metrics
    """
    import xgboost as xgb

    # ── Load model ────────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"M3 model not found at {MODEL_PATH}\n"
            "Run training first: python m3_classifier/train_xgb.py"
        )

    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    logger.info(f"[EVAL] Model loaded from {MODEL_PATH}")

    meta = {}
    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
        logger.info(f"[EVAL] Metadata: F2={meta.get('test_f2','?')}, TSS={meta.get('test_tss','?')}")

    # Use threshold from metadata unless overridden
    if threshold is None:
        threshold = float(meta.get("optimal_threshold", 0.5))
    logger.info(f"[EVAL] Using threshold: {threshold}")

    # ── Load test set (same chronological split as training) ──────────────────
    df = load_sharp_dataset()
    X, y, _ = build_feature_matrix(df, save_scaler=False)

    n = len(X)
    train_end = int(n * CONFIG["train_ratio"])
    val_end   = int(n * (CONFIG["train_ratio"] + CONFIG["val_ratio"]))

    X_test = X[val_end:]
    y_test = y[val_end:]

    logger.info(f"[EVAL] Test set: {len(X_test)} samples, "
                f"{y_test.sum()} positive ({y_test.mean()*100:.2f}%)")

    # ── Predict ───────────────────────────────────────────────────────────────
    dtest = xgb.DMatrix(X_test, feature_names=FEATURE_COLS)
    probs = model.predict(dtest)
    y_pred = (probs >= threshold).astype(int)

    # ── Compute metrics ───────────────────────────────────────────────────────
    tp, tn, fp, fn = confusion_components(y_test, y_pred)

    tss = compute_tss(tp, tn, fp, fn)
    f2  = compute_fbeta(tp, fp, fn, beta=2.0)
    f1  = compute_fbeta(tp, fp, fn, beta=1.0)
    hss = compute_hss(tp, tn, fp, fn)

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_test, probs)) if y_test.sum() > 0 else 0.0
    except Exception:
        auc = 0.0

    # ── Threshold sweep ───────────────────────────────────────────────────────
    sweep_df = threshold_sweep(probs, y_test)
    best_tss_row = sweep_df.loc[sweep_df["TSS"].idxmax()]
    best_f2_row  = sweep_df.loc[sweep_df["F2"].idxmax()]

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ICARUS-X M3 Sentinel — Evaluation Report")
    print("=" * 65)
    print(f"  Test samples:   {len(X_test):,}")
    print(f"  Positives:      {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    print(f"  Threshold:      {threshold:.2f}")
    print()
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print()

    tss_pass  = "✅" if tss  > 0.65 else "❌"
    f2_pass   = "✅" if f2   > 0.70 else "❌"

    print(f"  TSS  (target >0.65): {tss:.4f}  {tss_pass}")
    print(f"  F2   (target >0.70): {f2:.4f}  {f2_pass}")
    print(f"  F1:                  {f1:.4f}")
    print(f"  HSS:                 {hss:.4f}")
    print(f"  ROC-AUC:             {auc:.4f}")
    print(f"  Precision:           {precision:.4f}")
    print(f"  Recall:              {recall:.4f}")
    print()
    print(f"  Best TSS @ threshold={best_tss_row['threshold']:.2f}: "
          f"TSS={best_tss_row['TSS']:.4f}")
    print(f"  Best F2  @ threshold={best_f2_row['threshold']:.2f}:  "
          f"F2={best_f2_row['F2']:.4f}")
    print("=" * 65)

    if tss < 0.65 or f2 < 0.70:
        print("\n⚠️  One or more metrics below target.")
        print("   Suggestions:")
        if y_test.mean() < 0.01:
            print("   - Very low positive rate in test set — download more data (jsoc source)")
        if recall < 0.5:
            print(f"   - Low recall ({recall:.2f}): try threshold={best_f2_row['threshold']:.2f}")
        print("   - Retrain with more Optuna trials: python m3_classifier/train_xgb.py --trials 100")

    results = {
        "threshold":  threshold,
        "n_test":     len(X_test),
        "n_pos":      int(y_test.sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "TSS":       round(tss, 4),
        "F2":        round(f2,  4),
        "F1":        round(f1,  4),
        "HSS":       round(hss, 4),
        "AUC":       round(auc, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "best_tss_threshold": float(best_tss_row["threshold"]),
        "best_f2_threshold":  float(best_f2_row["threshold"]),
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate M3 flare classifier")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Probability threshold (default: from training metadata)")
    args = parser.parse_args()
    evaluate(args.threshold)
