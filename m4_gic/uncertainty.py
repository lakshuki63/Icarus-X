"""
ICARUS-X — M4 GIC: MC Dropout Uncertainty Network

Small neural network that maps Kp + solar wind features → GIC
with Monte Carlo Dropout for uncertainty quantification.
Provides p5/p95 percentile bounds on GIC estimates.

Inputs:  Kp + contextual features (Bz, speed, density)
Outputs: GIC distribution (mean, p5, p95)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
from loguru import logger


class GICUncertaintyNet(nn.Module):
    """Small MLP with permanent dropout for MC uncertainty."""

    def __init__(self, input_size: int = 4, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Softplus(),  # GIC is always positive
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns GIC estimate."""
        return self.net(x).squeeze(-1)


def mc_dropout_predict(
    model: GICUncertaintyNet,
    x: torch.Tensor,
    n_samples: int = 100,
) -> Dict[str, float]:
    """Run MC Dropout inference for uncertainty bounds."""
    model.train()  # Keep dropout active

    preds = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.item())

    preds = np.array(preds)
    model.eval()

    return {
        "gic_mean": round(float(preds.mean()), 2),
        "gic_std": round(float(preds.std()), 2),
        "gic_p5": round(float(np.percentile(preds, 5)), 2),
        "gic_p95": round(float(np.percentile(preds, 95)), 2),
    }


def estimate_gic_uncertainty(
    kp: float,
    bz: float = 0.0,
    speed: float = 400.0,
    density: float = 5.0,
) -> Dict[str, float]:
    """
    Estimate GIC with uncertainty using empirical formula + noise model.
    This is the fallback when the neural net isn't trained.
    """
    from m4_gic.gic_model import kp_to_gic

    base_gic = kp_to_gic(kp)

    # Add physics-informed uncertainty based on solar wind conditions
    bz_factor = max(0, -bz) / 10.0  # Southward Bz amplifies GIC
    speed_factor = max(0, speed - 400) / 200.0  # Fast wind amplifies

    # Uncertainty grows with Kp (more uncertain at higher activity)
    uncertainty_pct = 0.15 + 0.05 * kp + 0.1 * bz_factor

    rng = np.random.RandomState()
    samples = rng.normal(base_gic * (1 + 0.1 * bz_factor + 0.05 * speed_factor),
                          base_gic * uncertainty_pct, 100)
    samples = np.maximum(samples, 0)

    return {
        "gic_mean": round(float(samples.mean()), 2),
        "gic_std": round(float(samples.std()), 2),
        "gic_p5": round(float(np.percentile(samples, 5)), 2),
        "gic_p95": round(float(np.percentile(samples, 95)), 2),
    }
