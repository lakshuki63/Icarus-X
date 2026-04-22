"""
ICARUS-X — M3 Classifier: Feature Engineering

Builds handcrafted features from solar wind time windows for
XGBoost storm classification. Features capture statistical
summaries over 3h, 6h, and 12h windows.

Inputs:  Solar wind time series (Bz, speed, density, Kp)
Outputs: Feature matrix for XGBoost
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from loguru import logger


# ── Feature definitions ──────────────────────────────────
WINDOW_HOURS = [3, 6, 12]

BASE_COLS = ["bz_gsm", "speed", "density", "bt", "kp_value"]


def compute_window_features(
    df: pd.DataFrame,
    idx: int,
    window_hrs: int,
) -> Dict[str, float]:
    """Compute statistical features over a time window ending at idx."""
    start = max(0, idx - window_hrs)
    window = df.iloc[start:idx + 1]

    features = {}
    suffix = f"_{window_hrs}hr"

    for col in BASE_COLS:
        if col not in window.columns:
            continue
        vals = window[col].dropna()
        if len(vals) == 0:
            features[f"mean_{col}{suffix}"] = 0.0
            features[f"std_{col}{suffix}"] = 0.0
            features[f"min_{col}{suffix}"] = 0.0
            features[f"max_{col}{suffix}"] = 0.0
            continue

        features[f"mean_{col}{suffix}"] = float(vals.mean())
        features[f"std_{col}{suffix}"] = float(vals.std())
        features[f"min_{col}{suffix}"] = float(vals.min())
        features[f"max_{col}{suffix}"] = float(vals.max())

    # Derived features
    if "bz_gsm" in window.columns:
        bz = window["bz_gsm"].dropna()
        if len(bz) > 0:
            features[f"bz_southward_frac{suffix}"] = float((bz < 0).mean())
            features[f"bz_min_abs{suffix}"] = float(bz.abs().max())

    if "speed" in window.columns and "density" in window.columns:
        v = window["speed"].dropna()
        d = window["density"].dropna()
        if len(v) > 0 and len(d) > 0:
            # Dynamic pressure proxy: ρ * V²
            features[f"dyn_pressure{suffix}"] = float((d.mean() * (v.mean() ** 2)) / 1e6)

    return features


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build full feature matrix from time series DataFrame."""
    logger.info("🔧 Building feature matrix...")

    all_features = []
    max_window = max(WINDOW_HOURS)

    for idx in range(max_window, len(df)):
        row_features = {"timestamp": df.iloc[idx].get("timestamp")}

        for wh in WINDOW_HOURS:
            feats = compute_window_features(df, idx, wh)
            row_features.update(feats)

        # Current values
        for col in BASE_COLS:
            if col in df.columns:
                row_features[f"current_{col}"] = float(df.iloc[idx][col])

        all_features.append(row_features)

    feature_df = pd.DataFrame(all_features)
    logger.info(f"   ✅ Feature matrix: {feature_df.shape}")
    return feature_df


def get_feature_names() -> List[str]:
    """Return ordered list of feature column names (excluding timestamp)."""
    names = []
    for wh in WINDOW_HOURS:
        suffix = f"_{wh}hr"
        for col in BASE_COLS:
            names.extend([
                f"mean_{col}{suffix}",
                f"std_{col}{suffix}",
                f"min_{col}{suffix}",
                f"max_{col}{suffix}",
            ])
        names.extend([
            f"bz_southward_frac{suffix}",
            f"bz_min_abs{suffix}",
            f"dyn_pressure{suffix}",
        ])
    for col in BASE_COLS:
        names.append(f"current_{col}")
    return names


def build_realtime_features(readings: List[Dict]) -> Dict[str, float]:
    """Build features from a list of real-time solar wind readings."""
    if not readings:
        return {name: 0.0 for name in get_feature_names()}

    df = pd.DataFrame(readings)
    for col in BASE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    features = {}
    idx = len(df) - 1

    for wh in WINDOW_HOURS:
        feats = compute_window_features(df, idx, wh)
        features.update(feats)

    for col in BASE_COLS:
        if col in df.columns:
            features[f"current_{col}"] = float(df.iloc[-1][col])

    return features
