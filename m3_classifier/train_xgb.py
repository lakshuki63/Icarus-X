"""
ICARUS-X — M3 Sentinel: XGBoost Flare Classifier Training (INDEPENDENT MODULE)

Trains a binary XGBoost classifier to predict M/X-class solar flares
from 6 SHARP photospheric parameters. Fully independent of M1, M2, M4.

Pipeline:
  1. Load SHARP dataset (data/sharp_flare_dataset.csv)
  2. Log1p transform (see features.py)
  3. Chronological train/val/test split (no shuffle — time series)
  4. SMOTE oversampling on training set ONLY
  5. Optuna HPO (50 trials, maximise F2-score on validation set)
  6. Retrain best model on train+val, evaluate on held-out test set
  7. SHAP explainability
  8. Save model + metadata to models/

Inputs:  data/sharp_flare_dataset.csv
Outputs: models/xgb_flare_sentinel.json
         models/xgb_flare_sentinel_meta.json
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# ── Project root — ZERO M2 IMPORTS ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from m3_classifier.features import (
    FEATURE_COLS,
    FEATURE_DESCRIPTIONS,
    build_feature_matrix,
    load_sharp_dataset,
)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "model_path":    PROJECT_ROOT / "models" / "xgb_flare_sentinel.json",
    "meta_path":     PROJECT_ROOT / "models" / "xgb_flare_sentinel_meta.json",
    "optuna_trials": 50,
    "optuna_seed":   42,
    "f2_beta":       2.0,    # F2 weights recall 2× more than precision
    "focal_gamma":   2.0,    # Focal loss focusing parameter
    "focal_alpha":   0.25,   # Class weight for minority (flare) class
    "train_ratio":   0.70,
    "val_ratio":     0.15,
    # test_ratio = 0.15 (remainder)
    "smote_k":       5,      # SMOTE k_neighbors (reduced to 3 if class too small)
    "random_seed":   42,
}

(PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)


# ── Focal Loss (Binary) ───────────────────────────────────────────────────────
def focal_loss_objective(
    y_pred: np.ndarray,
    dtrain: xgb.DMatrix,
    gamma: float = CONFIG["focal_gamma"],
    alpha: float = CONFIG["focal_alpha"],
):
    """
    Binary focal loss objective for XGBoost.

    FL(p) = -alpha * (1-p)^gamma * log(p)   for positive class
    FL(p) = -(1-alpha) * p^gamma * log(1-p) for negative class

    Focuses training on hard, misclassified examples.
    Critical for the severe class imbalance (~2–5% flare rate).

    Args:
        y_pred: Raw model scores (pre-sigmoid)
        dtrain: XGBoost DMatrix with labels
        gamma: Focusing parameter (2.0 = standard focal loss)
        alpha: Balance parameter for minority class

    Returns:
        (grad, hess) tuple — gradient and hessian of focal loss
    """
    labels = dtrain.get_label()

    # Sigmoid of raw scores → probabilities
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1.0 - 1e-7)

    # Per-class alpha weighting
    alpha_t = np.where(labels == 1, alpha, 1.0 - alpha)

    # Focal weight: (1 - p_t)^gamma for correct class
    p_t = np.where(labels == 1, p, 1.0 - p)
    focal_weight = (1.0 - p_t) ** gamma

    # Gradient
    grad = alpha_t * focal_weight * (p - labels)

    # Hessian (second derivative approximation)
    hess = alpha_t * focal_weight * p * (1.0 - p)
    hess = np.maximum(hess, 1e-7)  # Prevent zero hessian

    return grad, hess


def focal_loss_eval(
    y_pred: np.ndarray,
    dtrain: xgb.DMatrix,
    gamma: float = CONFIG["focal_gamma"],
    alpha: float = CONFIG["focal_alpha"],
) -> Tuple[str, float]:
    """
    Focal loss eval metric (for evals= reporting).

    Returns:
        ('focal_loss', value) — lower is better
    """
    labels = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1.0 - 1e-7)

    alpha_t = np.where(labels == 1, alpha, 1.0 - alpha)
    p_t = np.where(labels == 1, p, 1.0 - p)
    focal_weight = (1.0 - p_t) ** gamma
    loss = -alpha_t * focal_weight * np.log(p_t)

    return "focal_loss", float(loss.mean())


# ── F2-score for Optuna ───────────────────────────────────────────────────────
def compute_f2(
    model: xgb.Booster,
    dmat: xgb.DMatrix,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Compute F2-score (beta=2) on a dataset.

    F2 weights recall 2× more than precision — correct for rare flare events
    where missing a real flare is worse than a false alarm.

    Args:
        model: Trained XGBoost Booster
        dmat: DMatrix for prediction
        y_true: True binary labels
        threshold: Probability threshold for positive class

    Returns:
        F2-score in [0, 1]
    """
    probs = model.predict(dmat)
    y_pred = (probs >= threshold).astype(int)
    return float(fbeta_score(y_true, y_pred, beta=CONFIG["f2_beta"], zero_division=0))


def find_optimal_threshold(
    model: xgb.Booster,
    dmat: xgb.DMatrix,
    y_true: np.ndarray,
) -> Tuple[float, float]:
    """
    Find probability threshold that maximises F2-score on validation set.

    Args:
        model: Trained XGBoost Booster
        dmat: Validation DMatrix
        y_true: Validation labels

    Returns:
        (best_threshold, best_f2)
    """
    probs = model.predict(dmat)
    best_t, best_f2 = 0.5, 0.0

    for t in np.linspace(0.1, 0.9, 81):
        y_pred = (probs >= t).astype(int)
        f2 = float(fbeta_score(y_true, y_pred, beta=CONFIG["f2_beta"], zero_division=0))
        if f2 > best_f2:
            best_f2 = f2
            best_t = t

    logger.info(f"  Optimal threshold: {best_t:.2f}  (F2={best_f2:.4f})")
    return float(best_t), float(best_f2)


# ── Data splitting ────────────────────────────────────────────────────────────
def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """
    Chronological train/val/test split — NO random shuffle.

    Time series data must not be shuffled: future data cannot be used
    to predict the past. The dataset is assumed sorted by timestamp.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    train_end = int(n * CONFIG["train_ratio"])
    val_end   = int(n * (CONFIG["train_ratio"] + CONFIG["val_ratio"]))

    X_train, y_train = X[:train_end],       y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    logger.info(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    for split_name, yy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        pos = yy.sum()
        logger.info(f"    {split_name} positives: {pos} ({pos/max(1,len(yy))*100:.2f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to the TRAINING SET ONLY.

    SMOTE is never applied to validation or test sets — that would leak
    synthetic minority samples into evaluation, inflating metrics.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        (X_resampled, y_resampled)
    """
    n_minority = int(y_train.sum())
    if n_minority < 2:
        logger.warning(f"  [!] Only {n_minority} positive samples in train set — SMOTE skipped")
        return X_train, y_train

    k = min(CONFIG["smote_k"], n_minority - 1)
    if k < 1:
        logger.warning(f"  [!] Too few minority samples for SMOTE (need ≥2) — skipped")
        return X_train, y_train

    try:
        smote = SMOTE(random_state=CONFIG["random_seed"], k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(f"  SMOTE: {len(X_train)} → {len(X_res)} samples "
                    f"(minority: {n_minority} → {(y_res==1).sum()})")
        return X_res, y_res
    except Exception as e:
        logger.warning(f"  [!] SMOTE failed: {e} — using original training data")
        return X_train, y_train


# ── Optuna HPO ────────────────────────────────────────────────────────────────
def run_optuna_hpo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    n_trials: int = CONFIG["optuna_trials"],
) -> Dict:
    """
    Optuna hyperparameter optimisation — maximises F2-score on validation set.

    Args:
        X_train: SMOTE-resampled training features
        y_train: SMOTE-resampled training labels
        X_val:   Validation features (not resampled)
        y_val:   Validation labels
        n_trials: Number of Optuna trials

    Returns:
        Dict of best hyperparameters
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("[!] optuna not installed — using default hyperparameters. "
                       "Run: pip install optuna>=3.6.0")
        return _default_params()

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=FEATURE_COLS)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "eta":              trial.suggest_float("eta", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "tree_method":      "hist",
            "verbosity":        0,
        }
        n_rounds = trial.suggest_int("n_estimators", 100, 600)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            obj=lambda pred, dtrain: focal_loss_objective(pred, dtrain),
            custom_metric=focal_loss_eval,
            verbose_eval=False,
        )

        f2 = compute_f2(model, dval, y_val)
        return f2

    study = optuna.create_study(
        direction="maximize",
        study_name="m3_flare_f2",
        sampler=optuna.samplers.TPESampler(seed=CONFIG["optuna_seed"]),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(f"  [OPTUNA] Best F2={study.best_value:.4f} after {n_trials} trials")
    logger.info(f"  Best params: {best}")
    return best


def _default_params() -> Dict:
    """Fallback hyperparameters when Optuna is unavailable."""
    return {
        "max_depth": 6,
        "eta": 0.05,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 300,
        "tree_method": "hist",
        "verbosity": 0,
    }


# ── SHAP ──────────────────────────────────────────────────────────────────────
def compute_shap_summary(
    model: xgb.Booster,
    X_test: np.ndarray,
) -> List[Dict]:
    """
    Compute mean |SHAP| values for feature importance.

    Args:
        model: Trained XGBoost Booster
        X_test: Test set features

    Returns:
        List of {name, mean_abs_shap} dicts, sorted by importance
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        mean_abs = np.abs(shap_values).mean(axis=0)
        results = [
            {"name": FEATURE_DESCRIPTIONS[feat], "mean_abs_shap": round(float(v), 6)}
            for feat, v in zip(FEATURE_COLS, mean_abs)
        ]
        results.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        return results
    except Exception as e:
        logger.warning(f"  [!] SHAP computation failed: {e}")
        # Fallback: use XGBoost built-in gain importance
        scores = model.get_score(importance_type="gain")
        total = sum(scores.values()) or 1.0
        return [
            {"name": FEATURE_DESCRIPTIONS.get(k, k), "mean_abs_shap": round(v / total, 6)}
            for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]


# ── Main training function ────────────────────────────────────────────────────
def train_classifier(
    dataset_path: Optional[Path] = None,
    n_trials: int = CONFIG["optuna_trials"],
) -> None:
    """
    Full M3 training pipeline: load → split → SMOTE → HPO → train → evaluate → save.

    Args:
        dataset_path: Path to sharp_flare_dataset.csv (defaults to data/)
        n_trials: Number of Optuna HPO trials
    """
    logger.info("=" * 60)
    logger.info("ICARUS-X M3 Sentinel — XGBoost Flare Classifier Training")
    logger.info("=" * 60)

    # ── 1. Load & transform ───────────────────────────────────────────────────
    df = load_sharp_dataset(dataset_path)
    X, y, _ = build_feature_matrix(df, save_scaler=True)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    logger.info(f"[DATA] Total: {len(X)} | Positive (M/X flare): {n_pos} | "
                f"Negative: {n_neg} | Rate: {n_pos/len(X)*100:.2f}%")

    # ── 2. Chronological split ────────────────────────────────────────────────
    logger.info("[SPLIT] Chronological train/val/test split (no shuffle)...")
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y)

    # ── 3. SMOTE (training only) ──────────────────────────────────────────────
    logger.info("[SMOTE] Oversampling minority class in training set only...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # ── 4. Optuna HPO ─────────────────────────────────────────────────────────
    logger.info(f"[HPO] Optuna hyperparameter search ({n_trials} trials)...")
    best_params = run_optuna_hpo(X_train_res, y_train_res, X_val, y_val, n_trials)

    n_estimators = best_params.pop("n_estimators", 300)

    # ── 5. Final training on train+val ────────────────────────────────────────
    logger.info("[TRAIN] Retraining on train+val with best hyperparameters...")
    X_trainval = np.vstack([X_train_res, X_val])
    y_trainval = np.concatenate([y_train_res, y_val])

    dtrain_final = xgb.DMatrix(X_trainval, label=y_trainval, feature_names=FEATURE_COLS)
    dval_dm      = xgb.DMatrix(X_val,      label=y_val,      feature_names=FEATURE_COLS)
    dtest_dm     = xgb.DMatrix(X_test,     label=y_test,     feature_names=FEATURE_COLS)

    model = xgb.train(
        best_params,
        dtrain_final,
        num_boost_round=n_estimators,
        evals=[(dtrain_final, "train"), (dval_dm, "val")],
        obj=lambda pred, dtrain: focal_loss_objective(pred, dtrain),
        custom_metric=focal_loss_eval,
        verbose_eval=50,
    )

    # ── 6. Find optimal threshold ─────────────────────────────────────────────
    logger.info("[THRESHOLD] Finding optimal F2 threshold on validation set...")
    best_threshold, best_val_f2 = find_optimal_threshold(model, dval_dm, y_val)

    # ── 7. Test set evaluation ────────────────────────────────────────────────
    logger.info("[EVAL] Final evaluation on held-out test set...")
    test_probs = model.predict(dtest_dm)
    y_pred_test = (test_probs >= best_threshold).astype(int)

    test_f2  = float(fbeta_score(y_test, y_pred_test, beta=2.0, zero_division=0))
    test_auc = float(roc_auc_score(y_test, test_probs)) if y_test.sum() > 0 else 0.0

    # TSS (True Skill Statistic) = sensitivity + specificity - 1
    cm = confusion_matrix(y_test, y_pred_test)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tss = float((tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1)) - 1.0)
    else:
        tss = 0.0

    report = classification_report(
        y_test, y_pred_test,
        target_names=["No Flare (0)", "M/X Flare (1)"],
        zero_division=0,
    )

    logger.info(f"\n[RESULTS] Test Set Evaluation:")
    logger.info(f"  F2-score:  {test_f2:.4f}  (target: >0.70)")
    logger.info(f"  TSS:       {tss:.4f}  (target: >0.65)")
    logger.info(f"  ROC-AUC:   {test_auc:.4f}")
    logger.info(f"  Threshold: {best_threshold:.2f}")
    logger.info(f"\n{report}")

    if test_f2 < 0.50:
        logger.warning(
            "[!] F2-score below 0.50 — model may need more data or different HPO.\n"
            "    Try: python m3_classifier/data_download.py --source jsoc\n"
            "    Then retrain with more trials: --trials 100"
        )

    # ── 8. SHAP ───────────────────────────────────────────────────────────────
    logger.info("[SHAP] Computing feature importance...")
    shap_summary = compute_shap_summary(model, X_test)
    logger.info("  Top features:")
    for feat_info in shap_summary[:3]:
        logger.info(f"    {feat_info['name']}: {feat_info['mean_abs_shap']:.6f}")

    # ── 9. Save model + metadata ──────────────────────────────────────────────
    model_path = CONFIG["model_path"]
    meta_path  = CONFIG["meta_path"]

    model.save_model(str(model_path))
    logger.info(f"[SAVE] XGBoost model → {model_path}")

    meta = {
        "feature_names":       FEATURE_COLS,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "optimal_threshold":   best_threshold,
        "val_f2":              round(best_val_f2, 4),
        "test_f2":             round(test_f2, 4),
        "test_tss":            round(tss, 4),
        "test_auc":            round(test_auc, 4),
        "n_train":             len(X_train_res),
        "n_val":               len(X_val),
        "n_test":              len(X_test),
        "n_positives_train":   int(y_train_res.sum()),
        "focal_gamma":         CONFIG["focal_gamma"],
        "focal_alpha":         CONFIG["focal_alpha"],
        "best_hparams":        {**best_params, "n_estimators": n_estimators},
        "shap_summary":        shap_summary,
        "transform":           "log1p after physical clip",
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"[SAVE] Metadata → {meta_path}")

    logger.info("\n[DONE] M3 training complete.")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  F2={test_f2:.4f} | TSS={tss:.4f} | AUC={test_auc:.4f}")
    logger.info("  Next: python m3_classifier/infer.py")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ICARUS-X M3 XGBoost flare classifier")
    parser.add_argument("--trials", type=int, default=CONFIG["optuna_trials"],
                        help=f"Optuna HPO trials (default: {CONFIG['optuna_trials']})")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to sharp_flare_dataset.csv (default: data/)")
    args = parser.parse_args()

    train_classifier(
        dataset_path=Path(args.data) if args.data else None,
        n_trials=args.trials,
    )
