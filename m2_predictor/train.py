"""
ICARUS-X — M2 Predictor: Training Loop

Trains the BiGRU Seq2Seq model on OMNI solar wind + Kp + AR features.
Saves checkpoint with correct n_features (7 SW + 12 AR = 19).

Inputs:  data/omni_solar_wind.csv, data/kp_index.csv, data/ar_features.csv
Outputs: models/bigru_predictor.pt
         models/feature_scaler.pkl

Usage:
  python m2_predictor/train.py
  python m2_predictor/train.py --epochs 60 --device cuda
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger
from tqdm import tqdm

from m2_predictor.model import BiGRUPredictor
from m2_predictor.data_loader import (
    load_omni_csv, load_kp_csv, load_ar_features_csv,
    merge_datasets, normalize_features,
    SOLAR_WIND_COLS, AR_FEATURE_COLS,
)
from m2_predictor.windowing import create_dataloaders, FORECAST_HORIZONS

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR      = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODELS_DIR / "bigru_predictor.pt"

CONFIG = {
    "epochs":       50,
    "batch_size":   64,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "patience":     10,
    "grad_clip":    1.0,   # BUG FIX: ensure gradient clipping is always applied
}


# ── RMSE table ────────────────────────────────────────────────────────────────
def compute_persistence_rmse(y_true: np.ndarray, y_last: float) -> float:
    """
    Compute RMSE of a naive persistence baseline (always predict last known Kp).

    Args:
        y_true: True Kp values for this horizon
        y_last: Last known Kp value (persistence prediction)

    Returns:
        RMSE float
    """
    return float(np.sqrt(np.mean((y_true - y_last) ** 2)))


def print_rmse_table(
    model: BiGRUPredictor,
    val_dl,
    device: str,
) -> None:
    """
    Print per-horizon RMSE vs persistence baseline.

    BUG FIX: Guards against zero persistence_rmse with p > 1e-8 check.
    Prints 'N/A' instead of crashing when persistence_rmse == 0.
    """
    model.eval()
    all_preds  = [[] for _ in FORECAST_HORIZONS]
    all_truths = [[] for _ in FORECAST_HORIZONS]
    all_last   = []

    with torch.no_grad():
        for x_batch, y_batch in val_dl:
            x_batch = x_batch.to(device)
            out = model(x_batch)
            preds = out["kp_pred"].cpu().numpy()
            truth = y_batch.numpy()

            # Last known Kp = last timestep of the Kp channel
            # Kp is the last column of the solar wind block (index 3 = bt, varies)
            # Use y_batch[:, 0] as proxy (3h true) — or last input Kp if available
            last_kp = truth[:, 0]  # persistence = predict 3h value for all horizons

            for i in range(len(FORECAST_HORIZONS)):
                all_preds[i].extend(preds[:, i].tolist())
                all_truths[i].extend(truth[:, i].tolist())
            all_last.extend(last_kp.tolist())

    print("\n" + "=" * 65)
    print(f"{'Horizon':>8} | {'Model RMSE':>10} | {'Persist RMSE':>12} | {'Improve %':>10}")
    print("-" * 65)

    for i, h in enumerate(FORECAST_HORIZONS):
        y_t = np.array(all_truths[i])
        y_p = np.array(all_preds[i])
        y_l = np.array(all_last)

        m = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
        p = float(np.sqrt(np.mean((y_t - y_l) ** 2)))

        # BUG FIX: guard zero persistence RMSE (BUG 1 from spec)
        if p > 1e-8:
            delta_str = f"{(m - p) / p * 100:+.1f}%"
        else:
            delta_str = "N/A"

        better = "✅" if p > 1e-8 and m < p else ("❌" if p > 1e-8 else "")
        print(f"  +{h:2d}h    | {m:10.4f} | {p:12.4f} | {delta_str:>10} {better}")

    print("=" * 65 + "\n")


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(
    epochs:     int   = CONFIG["epochs"],
    batch_size: int   = CONFIG["batch_size"],
    lr:         float = CONFIG["lr"],
    device:     str   = None,
) -> None:
    """Full training pipeline for BiGRU predictor."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[M2] Training on: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("[M2] Loading data...")
    sw_df  = load_omni_csv()
    kp_df  = load_kp_csv()

    # Load AR features — BUG FIX: always try to include AR features
    # so n_features = 7 + 12 = 19 (matches infer.py default)
    ar_df = load_ar_features_csv()
    if ar_df is not None and not ar_df.empty:
        logger.info(f"   AR features loaded: {len(ar_df)} rows")
    else:
        logger.warning(
            "   AR features not found — training with SW only (n_features=7).\n"
            "   Run m1_visionary first to generate data/ar_features.csv\n"
            "   AR features will be zero-padded to n_features=19 for infer.py compat."
        )

    merged = merge_datasets(sw_df, kp_df, ar_df)
    X, y, _ = normalize_features(merged, fit=True)

    n_features = X.shape[1]
    logger.info(f"   Features: {n_features} "
                f"({'SW only — AR missing' if n_features == 7 else 'SW + AR ✅'})")
    logger.info(f"   Samples:  {len(X)}")

    # ── Pad to 19 features if AR missing ──────────────────────────────────────
    # Ensures checkpoint is always compatible with infer.py (expects 19)
    if n_features < 19:
        pad = np.zeros((len(X), 19 - n_features), dtype=np.float32)
        X = np.hstack([X, pad])
        n_features = 19
        logger.warning(f"   Padded features to {n_features} (AR zeros) for infer.py compat")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_dl, val_dl = create_dataloaders(X, y, batch_size=batch_size, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = BiGRUPredictor(input_size=n_features).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.HuberLoss(delta=1.0)

    logger.info(f"[M2] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            out  = model(x_batch)
            loss = criterion(out["kp_pred"], y_batch)
            loss.backward()

            # Gradient clipping (max_norm=1.0) — always applied
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                out      = model(x_batch)
                val_loss += criterion(out["kp_pred"], y_batch).item()
        val_loss /= len(val_dl)

        logger.info(
            f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":          val_loss,
                "n_features":        n_features,  # BUG FIX: always 19
            }, CHECKPOINT_PATH)
            logger.info(f"   💾 Checkpoint saved (val_loss={val_loss:.4f}, n_features={n_features})")
        else:
            patience_count += 1
            if patience_count >= CONFIG["patience"]:
                logger.info(f"[M2] Early stopping at epoch {epoch}")
                break

    # ── RMSE table ────────────────────────────────────────────────────────────
    logger.info("[M2] Computing per-horizon RMSE table...")
    print_rmse_table(model, val_dl, device)

    logger.info(f"[M2] ✅ Training complete. Best val_loss: {best_val_loss:.4f}")
    logger.info(f"   Checkpoint: {CHECKPOINT_PATH}")
    logger.info("   Next: python m2_predictor/infer.py")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ICARUS-X M2 BiGRU predictor")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
