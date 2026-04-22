"""
ICARUS-X — M2 Predictor: Training Loop

Trains the BiGRU Seq2Seq model on OMNI + Kp data.
Supports: cosine LR schedule, early stopping, checkpoint saving.

Inputs:  Synthetic or real OMNI/Kp CSV data
Outputs: Trained model checkpoint at models/bigru_predictor.pt
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger
from tqdm import tqdm

from m2_predictor.model import BiGRUPredictor
from m2_predictor.data_loader import prepare_training_data
from m2_predictor.windowing import create_dataloaders

# ── Configuration ────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODELS_DIR / "bigru_predictor.pt"

EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # early stopping


def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    device: str = None,
) -> None:
    """Full training pipeline for BiGRU predictor."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️  Training on: {device}")

    # ── Load data ────────────────────────────────────────
    logger.info("📂 Loading training data...")
    X, y, df = prepare_training_data()
    n_features = X.shape[1]
    logger.info(f"   Features: {n_features}, Samples: {len(X)}")

    # ── Create dataloaders ───────────────────────────────
    train_dl, val_dl = create_dataloaders(X, y, batch_size=batch_size, num_workers=0)

    # ── Initialize model ─────────────────────────────────
    model = BiGRUPredictor(input_size=n_features).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.HuberLoss(delta=1.0)

    logger.info(f"📊 Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training loop ────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out["kp_pred"], y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                out = model(x_batch)
                val_loss += criterion(out["kp_pred"], y_batch).item()
        val_loss /= len(val_dl)

        logger.info(
            f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "n_features": n_features,
            }, CHECKPOINT_PATH)
            logger.info(f"   💾 Checkpoint saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"⏹️  Early stopping at epoch {epoch}")
                break

    logger.info(f"✅ Training complete. Best val_loss: {best_val_loss:.4f}")
    logger.info(f"   Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train_model()
