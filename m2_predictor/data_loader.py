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


def _parse_ar_timestamp(raw: str) -> Optional[pd.Timestamp]:
    """
    Parse timestamp from ar_features.csv using multiple regex patterns.

    BUG 4 FIX: parse_ar_timestamp() was returning None for many filename
    formats, making the AR feature DataFrame entirely empty and training
    with all-zero AR features.

    Patterns tried (in order):
      1. ISO format column: '2017-09-06T12:00:00'
      2. Filename-encoded: '2017-09-06T120000__magnetogram'
      3. Date only: '2017-09-06'
      4. Compact: '20170906_1200'
    """
    import re
    if not isinstance(raw, str):
        try:
            return pd.Timestamp(raw)
        except Exception:
            return None

    patterns = [
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',   # ISO with colons
        r'(\d{4}-\d{2}-\d{2}T\d{6})',                  # ISO compact time
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',     # space-separated
        r'(\d{4}-\d{2}-\d{2})',                          # date only
        r'(\d{8}_\d{4})',                                # compact YYYYMMDD_HHMM
        r'(\d{14})',                                      # pure numeric 14-digit
    ]
    for pat in patterns:
        m = re.search(pat, raw)
        if m:
            try:
                return pd.Timestamp(m.group(1).replace('T', ' ').replace('_', ' '))
            except Exception:
                continue
    return None


def load_ar_features_csv(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Load AR feature CSV produced by M1 Visionary export.

    BUG 4 FIX: Handles multiple timestamp formats with fallback patterns.
    Logs parse success rate — raises warning if <10% parse successfully.

    Args:
        path: Path to ar_features.csv (defaults to data/ar_features.csv)

    Returns:
        DataFrame with [timestamp, f0..f11] or None if file not found
    """
    if path is None:
        path = DATA_DIR / "ar_features.csv"

    if not path.exists():
        logger.warning(f"[DATA] AR features file not found: {path}")
        logger.warning("   Train M1 first: python m1_visionary/export_features.py")
        return None

    logger.info(f"[DATA] Loading AR features from {path}")
    df = pd.read_csv(path)
    total_rows = len(df)

    # Try direct timestamp column first
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        n_parsed = df["timestamp"].notna().sum()
    elif "filename" in df.columns:
        # Parse from filename column (M1 output format)
        df["timestamp"] = df["filename"].apply(_parse_ar_timestamp)
        n_parsed = df["timestamp"].notna().sum()
    else:
        logger.warning("[DATA] AR features: no 'timestamp' or 'filename' column found")
        # Assign sequential timestamps as fallback
        df["timestamp"] = pd.date_range("2017-01-01", periods=total_rows, freq="12min")
        n_parsed = total_rows

    parse_rate = n_parsed / max(total_rows, 1)
    logger.info(f"   Parsed {n_parsed}/{total_rows} timestamps ({parse_rate*100:.1f}%)")

    if parse_rate < 0.10:
        logger.warning(
            f"   [!] Only {parse_rate*100:.1f}% of AR timestamps parsed successfully.\n"
            "   Check ar_features.csv format — ensure it has 'timestamp' or 'filename' column.\n"
            "   AR features will be zero-padded in training."
        )
        return None

    # Drop rows without valid timestamps
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Ensure all AR feature columns exist
    for col in AR_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Check for degenerate features
    f0_zero_frac = (df["f0"] == 0).mean()
    if f0_zero_frac > 0.5:
        logger.warning(
            f"   [!] {f0_zero_frac*100:.0f}% of f0 (total flux) values are zero.\n"
            "   M1 Visionary may not be producing valid AR features."
        )

    logger.info(f"   [OK] AR features: {len(df)} rows, f0_mean={df['f0'].mean():.4f}")
    return df[['timestamp'] + AR_FEATURE_COLS]


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
    kp_path:   Optional[Path] = None,
    ar_path:   Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Full pipeline: load → merge → normalize → return X, y, df.
    Includes AR features so n_features = 7 + 12 = 19.
    """
    sw_df = load_omni_csv(omni_path)
    kp_df = load_kp_csv(kp_path)
    ar_df = load_ar_features_csv(ar_path)  # None if file not found
    merged = merge_datasets(sw_df, kp_df, ar_df)
    X, y, _ = normalize_features(merged, fit=True)
    return X, y, merged


if __name__ == "__main__":
    X, y, df = prepare_training_data()
    print(f"Features shape: {X.shape}")
    print(f"Target shape:   {y.shape}")
    print(f"Kp range:       {y.min():.1f} – {y.max():.1f}")
    print(f"Time range:     {df['timestamp'].min()} → {df['timestamp'].max()}")
