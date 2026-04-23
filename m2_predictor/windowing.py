"""
ICARUS-X — M2 Predictor: Sliding Window Generator

Creates sliding windows from time series data for Seq2Seq training.
Each window = 24 hours of history → predict 8 future Kp horizons.

Inputs:  Normalized X, y arrays from data_loader
Outputs: PyTorch Dataset yielding (input_seq, target_horizons)
"""

from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger

# ── Configuration ────────────────────────────────────────
INPUT_WINDOW_HRS = 24          # 24 hours of history
FORECAST_HORIZONS = [3, 6, 9, 12, 15, 18, 21, 24]  # hours ahead
MAX_HORIZON = max(FORECAST_HORIZONS)


class SolarWindDataset(Dataset):
    """Sliding window dataset for Bi-GRU Seq2Seq training."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        input_window: int = INPUT_WINDOW_HRS,
        horizons: List[int] = None,
    ):
        """
        Args:
            X: Feature array of shape (T, n_features)
            y: Target Kp array of shape (T,)
            input_window: Number of timesteps for input sequence
            horizons: List of forecast horizons in timesteps
        """
        self.horizons = horizons or FORECAST_HORIZONS
        self.input_window = input_window
        self.max_horizon = max(self.horizons)

        # Valid indices: need input_window before + max_horizon after
        self.n_samples = len(X) - input_window - self.max_horizon
        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough data: {len(X)} rows for window={input_window} + horizon={self.max_horizon}"
            )

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        logger.info(f"   ✅ Dataset: {self.n_samples} samples, window={input_window}h, horizons={self.horizons}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_sequence, target_horizons)."""
        start = idx
        end = idx + self.input_window

        # Input: (input_window, n_features)
        x_seq = self.X[start:end]

        # Target: Kp at each forecast horizon
        targets = torch.stack([self.y[end + h - 1] for h in self.horizons])

        return x_seq, targets


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    batch_size: int = 64,
    num_workers: int = 0,   # BUG FIX: default 0 — avoids Windows multiprocessing errors
) -> Tuple[DataLoader, DataLoader]:
    """Split data and create train/val DataLoaders."""
    import torch
    # pin_memory only helps with CUDA; causes warnings on CPU
    use_pin = torch.cuda.is_available()

    split_idx = int(len(X) * train_ratio)

    train_ds = SolarWindDataset(X[:split_idx], y[:split_idx])
    val_ds   = SolarWindDataset(X[split_idx:], y[split_idx:])

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )

    logger.info(f"   ✅ Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    return train_dl, val_dl
