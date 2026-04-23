"""
ICARUS-X — M2 Predictor: Evaluation + Ablation Study

Evaluates BiGRU model per-horizon RMSE and compares:
  1. Model WITH AR features (n_features=19)
  2. Persistence baseline (always predict current Kp)
  3. Climatology baseline (always predict mean Kp)

This ablation shows whether M1's AR features actually help M2.

Usage:
  python m2_predictor/evaluate.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from loguru import logger

from m2_predictor.model import BiGRUPredictor
from m2_predictor.data_loader import prepare_training_data
from m2_predictor.windowing import (
    SolarWindDataset, FORECAST_HORIZONS, create_dataloaders,
)

MODELS_DIR      = PROJECT_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "bigru_predictor.pt"
CONFIG = {"train_ratio": 0.8, "batch_size": 64}


def load_model(device: str) -> tuple:
    """Load BiGRU checkpoint. Returns (model, n_features)."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"BiGRU checkpoint not found at {CHECKPOINT_PATH}\n"
            "Run training first: python m2_predictor/train.py"
        )
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    n_features = ckpt.get("n_features", 19)
    model = BiGRUPredictor(input_size=n_features).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"[EVAL] BiGRU loaded (epoch={ckpt.get('epoch','?')}, n_features={n_features})")
    return model, n_features


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_model(
    model: BiGRUPredictor,
    val_dl,
    device: str,
) -> dict:
    """Collect per-horizon predictions on validation set."""
    all_preds  = [[] for _ in FORECAST_HORIZONS]
    all_truths = [[] for _ in FORECAST_HORIZONS]
    all_last_kp = []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_dl:
            x_batch = x_batch.to(device)
            out = model(x_batch)
            preds  = out["kp_pred"].cpu().numpy()
            truths = y_batch.numpy()

            for i in range(len(FORECAST_HORIZONS)):
                all_preds[i].extend(preds[:, i].tolist())
                all_truths[i].extend(truths[:, i].tolist())

            # Persistence = predict 3h true value for all horizons
            all_last_kp.extend(truths[:, 0].tolist())

    return {
        "preds":   [np.array(p) for p in all_preds],
        "truths":  [np.array(t) for t in all_truths],
        "last_kp": np.array(all_last_kp),
    }


def print_ablation_table(results: dict, mean_kp: float) -> None:
    """Print per-horizon ablation table."""
    preds   = results["preds"]
    truths  = results["truths"]
    last_kp = results["last_kp"]

    print("\n" + "=" * 80)
    print("ICARUS-X M2 Predictor — Ablation Evaluation")
    print("=" * 80)
    print(f"{'Horizon':>8} | {'Model RMSE':>10} | {'Persist RMSE':>12} | "
          f"{'Clim RMSE':>10} | {'vs Persist':>10} | {'AR helps?':>9}")
    print("-" * 80)

    n_better = 0
    for i, h in enumerate(FORECAST_HORIZONS):
        y_t = truths[i]
        y_p = preds[i]
        y_l = last_kp           # persistence
        y_c = np.full_like(y_t, mean_kp)  # climatology

        m_rmse = compute_rmse(y_t, y_p)
        p_rmse = compute_rmse(y_t, y_l)
        c_rmse = compute_rmse(y_t, y_c)

        # BUG 1 guard: avoid zero-division
        if p_rmse > 1e-8:
            delta_str = f"{(m_rmse - p_rmse) / p_rmse * 100:+.1f}%"
            ar_helps  = "✅ YES" if m_rmse < p_rmse else "❌ NO "
            if m_rmse < p_rmse:
                n_better += 1
        else:
            delta_str = "N/A"
            ar_helps  = "N/A"

        print(f"  +{h:2d}h    | {m_rmse:10.4f} | {p_rmse:12.4f} | "
              f"{c_rmse:10.4f} | {delta_str:>10} | {ar_helps}")

    print("=" * 80)
    print(f"\n  Model beats persistence in {n_better}/{len(FORECAST_HORIZONS)} horizons")
    print(f"  Target: 3h RMSE < 1.2 (check row +3h above)")

    if n_better == 0:
        logger.warning(
            "[EVAL] Model does NOT beat persistence at any horizon.\n"
            "  Possible causes:\n"
            "  1. AR features are all-zero (check data/ar_features.csv)\n"
            "  2. Too few training epochs — try: python m2_predictor/train.py --epochs 80\n"
            "  3. Training data is purely synthetic — use real OMNI data from OMNIWeb"
        )


def evaluate() -> None:
    """Full M2 evaluation pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[EVAL] Device: {device}")

    # Load data
    X, y, df = prepare_training_data()
    mean_kp = float(y.mean())
    logger.info(f"[EVAL] Kp mean={mean_kp:.3f}, range=[{y.min():.1f}, {y.max():.1f}]")

    # Chronological split
    n = len(X)
    split_idx = int(n * CONFIG["train_ratio"])

    # Pad to 19 if needed (same as train.py)
    if X.shape[1] < 19:
        X = np.hstack([X, np.zeros((n, 19 - X.shape[1]), dtype=np.float32)])

    X_val = X[split_idx:]
    y_val = y[split_idx:]

    val_ds = SolarWindDataset(X_val, y_val)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    # Load and evaluate model
    model, n_features = load_model(device)

    logger.info("[EVAL] Running evaluation...")
    results = evaluate_model(model, val_dl, device)
    print_ablation_table(results, mean_kp)

    # Check AR feature usage
    ar_non_zero = (X_val[:, 7:].abs() > 1e-6).any(axis=1).mean() if X_val.shape[1] > 7 else 0.0
    print(f"\n  AR features non-zero in {ar_non_zero*100:.1f}% of val samples")
    if ar_non_zero < 0.1:
        logger.warning(
            "[EVAL] Less than 10% of samples have non-zero AR features.\n"
            "  Run M1: python m1_visionary/export_features.py"
        )


if __name__ == "__main__":
    evaluate()
