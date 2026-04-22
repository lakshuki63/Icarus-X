"""
ICARUS-X — M2 Predictor: Data Loader

Loads OMNI solar wind CSV data and Kp index data for training the
Bi-GRU forecasting model. Handles missing values, normalization,
and merging of solar wind + AR features into unified time series.

Inputs:  CSV files from data/ directory (OMNI format)
Outputs: Normalized numpy arrays ready for windowing
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
import joblib

# ── Project root ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


# ── Solar wind feature columns (from OMNI / NOAA) ───────
SOLAR_WIND_COLS = [
    "bx_gsm", "by_gsm", "bz_gsm", "bt",
    "speed", "density", "temperature",
]

# ── AR feature columns (from M1 Visionary) ──────────────
AR_FEATURE_COLS = [f"f{i}" for i in range(12)]

# ── All input feature columns ───────────────────────────
ALL_FEATURE_COLS = SOLAR_WIND_COLS + AR_FEATURE_COLS

# ── Target column ───────────────────────────────────────
TARGET_COL = "kp_value"


def load_omni_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Load OMNI solar wind CSV, parse timestamps, handle missing values."""
    if path is None:
        path = DATA_DIR / "omni_solar_wind.csv"

    if not path.exists():
        logger.warning(f"[!] OMNI CSV not found at {path}, generating synthetic data")
        return _generate_synthetic_omni()

    logger.info(f"[DATA] Loading OMNI data from {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Replace fill values with NaN
    fill_values = [9999.99, 99999.9, 999.99, 99.99]
    for col in SOLAR_WIND_COLS:
        if col in df.columns:
            df[col] = df[col].replace(fill_values, np.nan)

    # Forward-fill then backward-fill missing values
    df[SOLAR_WIND_COLS] = df[SOLAR_WIND_COLS].ffill().bfill()

    logger.info(f"   [OK] Loaded {len(df)} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")
    return df


def load_kp_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Kp index CSV with timestamps."""
    if path is None:
        path = DATA_DIR / "kp_index.csv"

    if not path.exists():
        logger.warning(f"[!] Kp CSV not found at {path}, generating synthetic data")
        return _generate_synthetic_kp()

    logger.info(f"[DATA] Loading Kp index from {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"   [OK] Loaded {len(df)} Kp observations")
    return df


def merge_datasets(
    sw_df: pd.DataFrame,
    kp_df: pd.DataFrame,
    ar_df: Optional[pd.DataFrame] = None,
    freq: str = "1h",
) -> pd.DataFrame:
    """Merge solar wind, Kp, and AR features on a regular time grid."""
    # Resample solar wind to regular intervals
    sw_df = sw_df.set_index("timestamp").resample(freq).mean().reset_index()

    # Resample Kp (3-hourly → interpolate to 1-hourly)
    kp_df = kp_df.set_index("timestamp").resample(freq).interpolate(method="linear").reset_index()

    # Merge solar wind + Kp
    merged = pd.merge(sw_df, kp_df, on="timestamp", how="inner")

    # Merge AR features if provided
    if ar_df is not None and not ar_df.empty:
        ar_df = ar_df.set_index("timestamp").resample(freq).ffill().reset_index()
        merged = pd.merge(merged, ar_df, on="timestamp", how="left")

    # Fill AR feature NaNs with 0 (no active region detected)
    for col in AR_FEATURE_COLS:
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0.0)

    # Final cleanup
    merged = merged.dropna(subset=SOLAR_WIND_COLS + [TARGET_COL])
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"   [OK] Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
    return merged


def normalize_features(
    df: pd.DataFrame,
    fit: bool = True,
    scaler_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Normalize features using StandardScaler, return X, y arrays."""
    if scaler_path is None:
        scaler_path = MODELS_DIR / "feature_scaler.pkl"

    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"   [OK] Scaler fitted and saved to {scaler_path}")
    else:
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X)
        else:
            logger.warning("[!] No scaler found, using raw features")
            scaler = StandardScaler()

    return X, y, scaler


def _generate_synthetic_omni(n_hours: int = 8760) -> pd.DataFrame:
    """Generate 1 year of realistic synthetic solar wind data for training."""
    logger.info("[GEN] Generating synthetic OMNI data (1 year, hourly)")
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2017-01-01", periods=n_hours, freq="1h")

    # Base quiet conditions + occasional storm enhancements
    data = {
        "timestamp": timestamps,
        "bx_gsm": rng.normal(0, 3, n_hours),
        "by_gsm": rng.normal(0, 4, n_hours),
        "bz_gsm": rng.normal(0, 5, n_hours),  # Southward = negative
        "bt": np.abs(rng.normal(5, 3, n_hours)),
        "speed": rng.normal(400, 80, n_hours).clip(250, 900),
        "density": rng.lognormal(1.5, 0.8, n_hours).clip(0.5, 80),
        "temperature": rng.lognormal(11, 0.5, n_hours).clip(1e4, 1e6),
    }

    df = pd.DataFrame(data)

    # Inject Sept 2017 storm (hours 5880–5928 ≈ Sept 6–8)
    storm_start = 5880
    storm_end = min(storm_start + 48, n_hours)
    storm_idx = range(storm_start, storm_end)
    df.loc[storm_idx, "bz_gsm"] = rng.normal(-20, 5, len(storm_idx))  # Strong southward
    df.loc[storm_idx, "speed"] = rng.normal(700, 50, len(storm_idx))
    df.loc[storm_idx, "density"] = rng.lognormal(2.5, 0.5, len(storm_idx))
    df.loc[storm_idx, "bt"] = np.abs(rng.normal(25, 5, len(storm_idx)))

    return df


def _generate_synthetic_kp(n_hours: int = 8760) -> pd.DataFrame:
    """Generate 1 year of synthetic Kp index (3-hourly, interpolated to 1h)."""
    logger.info("[GEN] Generating synthetic Kp data (1 year)")
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2017-01-01", periods=n_hours, freq="1h")

    # Quiet Kp ~ 1-2, with occasional storms
    kp = rng.gamma(2, 0.8, n_hours).clip(0, 9).astype(np.float32)

    # Sept 2017 storm peak
    storm_start = 5880
    storm_end = min(storm_start + 48, n_hours)
    storm_profile = np.array([3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3])
    storm_extended = np.interp(
        np.linspace(0, len(storm_profile) - 1, storm_end - storm_start),
        np.arange(len(storm_profile)),
        storm_profile,
    )
    kp[storm_start:storm_end] = storm_extended

    return pd.DataFrame({"timestamp": timestamps, "kp_value": kp})


def prepare_training_data(
    omni_path: Optional[Path] = None,
    kp_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Full pipeline: load → merge → normalize → return X, y, df."""
    sw_df = load_omni_csv(omni_path)
    kp_df = load_kp_csv(kp_path)
    merged = merge_datasets(sw_df, kp_df)
    X, y, _ = normalize_features(merged, fit=True)
    return X, y, merged


if __name__ == "__main__":
    X, y, df = prepare_training_data()
    print(f"Features shape: {X.shape}")
    print(f"Target shape:   {y.shape}")
    print(f"Kp range:       {y.min():.1f} – {y.max():.1f}")
    print(f"Time range:     {df['timestamp'].min()} → {df['timestamp'].max()}")
