"""
ICARUS-X — M3 Classifier: Storm Event Builder

Builds labeled storm event dataset from Kp index history.
Maps Kp ranges to NOAA G-scale storm tiers for XGBoost training.

Inputs:  Kp time series
Outputs: DataFrame with G-tier labels (G0–G4)
"""

import numpy as np
import pandas as pd
from loguru import logger

# ── NOAA G-Scale Mapping ─────────────────────────────────
# G0: Kp 0–4   (below storm threshold)
# G1: Kp 5     (minor)
# G2: Kp 6     (moderate)
# G3: Kp 7     (strong)
# G4: Kp 8–9   (severe to extreme)
G_SCALE_THRESHOLDS = [
    (0, 5, 0),   # G0
    (5, 6, 1),   # G1
    (6, 7, 2),   # G2
    (7, 8, 3),   # G3
    (8, 10, 4),  # G4
]


def kp_to_g_tier(kp: float) -> int:
    """Convert Kp value to G-scale tier (0–4)."""
    for low, high, tier in G_SCALE_THRESHOLDS:
        if low <= kp < high:
            return tier
    return 4  # Kp >= 8 → G4


def build_storm_events(kp_series: pd.DataFrame) -> pd.DataFrame:
    """
    Build storm event dataset with G-tier labels.

    Args:
        kp_series: DataFrame with 'timestamp' and 'kp_value' columns

    Returns:
        DataFrame with added 'g_tier' column
    """
    df = kp_series.copy()
    df["g_tier"] = df["kp_value"].apply(kp_to_g_tier)

    # Log class distribution
    dist = df["g_tier"].value_counts().sort_index()
    logger.info("📊 G-tier distribution:")
    for tier, count in dist.items():
        pct = count / len(df) * 100
        logger.info(f"   G{tier}: {count:6d} ({pct:5.1f}%)")

    return df


def generate_synthetic_storm_events(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic storm event data with realistic class imbalance."""
    logger.info(f"🔧 Generating {n_samples} synthetic storm events")
    rng = np.random.RandomState(42)

    # Realistic class distribution (heavily imbalanced)
    class_probs = [0.65, 0.18, 0.10, 0.05, 0.02]
    tiers = rng.choice(5, size=n_samples, p=class_probs)

    # Generate Kp values consistent with tier
    kp_values = np.zeros(n_samples)
    for i, tier in enumerate(tiers):
        low, high, _ = G_SCALE_THRESHOLDS[tier]
        kp_values[i] = rng.uniform(low, high)

    timestamps = pd.date_range("2010-01-01", periods=n_samples, freq="3h")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "kp_value": kp_values,
        "g_tier": tiers,
    })

    return df
