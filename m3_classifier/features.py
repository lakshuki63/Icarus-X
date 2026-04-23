"""
ICARUS-X — M3 Sentinel: SHARP Feature Engineering (INDEPENDENT MODULE)

Loads and prepares the 6 SHARP photospheric features for solar flare
binary classification. This module has NO dependency on M1, M2, or M4.

Feature set (from Bobra & Couvidat 2015 / SWAN-SF):
  f0: TOTUSJH  — total unsigned current helicity          [A/m²]
  f1: TOTPOT   — total photospheric magnetic free energy  [erg/cm³]
  f2: TOTUSJZ  — total unsigned vertical current          [A]
  f3: ABSNJZH  — abs value of net current helicity        [G²/m]
  f4: R_VALUE  — flux near polarity inversion line        [Mx]
  f5: AREA_ACR — area of strong-field pixels              [μHem]

Inputs:  data/sharp_flare_dataset.csv (produced by data_download.py)
Outputs: (X, y) arrays for XGBoost training
         OR feature dict for single-sample inference
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from loguru import logger

# ── Project root — NO M2 IMPORTS ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_PATH  = DATA_DIR   / "sharp_flare_dataset.csv"
SCALER_PATH   = MODELS_DIR / "m3_sharp_scaler.pkl"

# ── Feature definitions ───────────────────────────────────────────────────────
FEATURE_COLS = ["f0", "f1", "f2", "f3", "f4", "f5"]

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "f0": "total_unsigned_current_helicity (TOTUSJH)",
    "f1": "free_energy_proxy (TOTPOT)",
    "f2": "total_unsigned_vertical_current (TOTUSJZ)",
    "f3": "lorentz_force_proxy (ABSNJZH)",
    "f4": "pil_flux (R_VALUE)",
    "f5": "field_area (AREA_ACR)",
}

# Physical value ranges for clipping (prevents extreme outliers from JSOC fills)
FEATURE_CLIP_BOUNDS: Dict[str, Tuple[float, float]] = {
    "f0": (0.0, 1e26),   # TOTUSJH
    "f1": (0.0, 1e28),   # TOTPOT
    "f2": (0.0, 1e16),   # TOTUSJZ
    "f3": (0.0, 1e18),   # ABSNJZH
    "f4": (0.0, 1e6),    # R_VALUE
    "f5": (0.0, 1e5),    # AREA_ACR
}

# M1 CNN feature (f0–f11) → M3 SHARP feature proxy mapping
# Used when running M3 at inference time without live JSOC data.
# M1 features are normalized outputs from the ARFeatureHead CNN.
# These mappings are approximations — documented explicitly.
M1_TO_M3_MAP: Dict[str, List[str]] = {
    "f0": ["f7", "f8"],        # current helicity ← M1 f7 (helicity) + f8 (R-value)
    "f1": ["f4", "f5"],        # free energy      ← M1 f4 (shear) + f5 (free energy proxy)
    "f2": ["f6", "f7"],        # vertical current ← M1 f6 (twist) + f7 (helicity)
    "f3": ["f3", "f9"],        # Lorentz force    ← M1 f3 (PIL gradient) + f9 (WL_SG)
    "f4": ["f8", "f10"],       # R_VALUE          ← M1 f8 (R-value) + f10 (effective field)
    "f5": ["f0", "f11"],       # AREA_ACR         ← M1 f0 (total flux) + f11 (complexity)
}
M1_TO_M3_WEIGHTS: Dict[str, List[float]] = {
    "f0": [0.6, 0.4],
    "f1": [0.5, 0.5],
    "f2": [0.55, 0.45],
    "f3": [0.6, 0.4],
    "f4": [0.65, 0.35],
    "f5": [0.7, 0.3],
}


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_sharp_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load SHARP flare dataset from CSV.

    Args:
        path: Path to sharp_flare_dataset.csv (defaults to data/)

    Returns:
        DataFrame with columns [timestamp, label, f0..f5]

    Raises:
        FileNotFoundError: If dataset file does not exist
    """
    if path is None:
        path = DATASET_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"M3 training data not found at {DATASET_PATH}\n"
            "Run first: python m3_classifier/data_download.py --source donki"
        )

    logger.info(f"[M3] Loading SHARP dataset from: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)

    # Verify required columns
    missing = [c for c in (["timestamp", "label"] + FEATURE_COLS) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            "Re-run: python m3_classifier/data_download.py"
        )

    logger.info(f"  Loaded: {len(df)} rows, {df['label'].sum()} positives "
                f"({df['label'].mean()*100:.2f}% positive rate)")
    return df


# ── Feature transforms ────────────────────────────────────────────────────────
def apply_log1p_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transform to all SHARP features.

    SHARP values span many orders of magnitude (e.g., TOTUSJH: 1e18–1e24).
    log1p compresses the range and makes XGBoost splits more effective.

    Args:
        df: DataFrame with f0..f5 columns

    Returns:
        DataFrame with log1p-transformed features (in-place copy)
    """
    df = df.copy()
    for feat in FEATURE_COLS:
        if feat in df.columns:
            # Clip first to remove JSOC fill-value artifacts
            lo, hi = FEATURE_CLIP_BOUNDS[feat]
            df[feat] = np.clip(df[feat].fillna(0.0), lo, hi)
            df[feat] = np.log1p(df[feat])
    return df


def build_feature_matrix(
    df: Optional[pd.DataFrame] = None,
    path: Optional[Path] = None,
    save_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build (X, y) arrays from SHARP dataset for XGBoost training.

    Applies: clip → log1p → returns raw arrays (XGBoost handles its own scaling).

    Args:
        df: Pre-loaded DataFrame (if None, loads from path/default)
        path: Path to CSV (used only if df is None)
        save_scaler: If True, saves feature stats to models/m3_sharp_scaler.pkl

    Returns:
        X: (N, 6) float32 feature array
        y: (N,)  int32 label array (0/1)
        feature_names: list of 6 feature name strings
    """
    if df is None:
        df = load_sharp_dataset(path)

    df = apply_log1p_transform(df)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)

    # Sanity checks
    assert X.shape[1] == 6, f"Expected 6 features, got {X.shape[1]}"
    assert len(np.unique(y)) >= 2, "Labels must contain both classes (0 and 1)"

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    logger.info(f"[M3] Feature matrix: {X.shape} | Positive: {n_pos} | Negative: {n_neg}")

    # Check for degenerate features (all-zero after log1p → likely missing data)
    for i, feat in enumerate(FEATURE_COLS):
        zero_frac = (X[:, i] == 0).mean()
        if zero_frac > 0.5:
            logger.warning(
                f"  [!] {feat} ({FEATURE_DESCRIPTIONS[feat]}): "
                f"{zero_frac*100:.0f}% zero after log1p — "
                "check if this SHARP parameter loaded correctly"
            )

    if save_scaler:
        scaler_meta = {
            "feature_names": FEATURE_COLS,
            "feature_descriptions": FEATURE_DESCRIPTIONS,
            "transform": "log1p after clip",
            "clip_bounds": FEATURE_CLIP_BOUNDS,
            "train_mean": X.mean(axis=0).tolist(),
            "train_std":  X.std(axis=0).tolist(),
            "n_samples": len(X),
            "n_positives": int(n_pos),
        }
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler_meta, SCALER_PATH)
        logger.info(f"  Scaler metadata saved: {SCALER_PATH}")

    return X, y, FEATURE_COLS


# ── Inference feature preparation ─────────────────────────────────────────────
def prepare_inference_features(
    feature_dict: Dict[str, float],
) -> np.ndarray:
    """
    Prepare a single feature dict for M3 inference.

    Applies the same clip → log1p transform used during training.

    Args:
        feature_dict: Dict with keys f0..f5, values in physical units
                      (raw SHARP values OR M1-mapped proxy values)

    Returns:
        (1, 6) float32 array ready for XGBoost predict
    """
    row = []
    for feat in FEATURE_COLS:
        val = float(feature_dict.get(feat, 0.0))
        lo, hi = FEATURE_CLIP_BOUNDS[feat]
        val = float(np.clip(val, lo, hi))
        val = float(np.log1p(val))
        row.append(val)

    x = np.array(row, dtype=np.float32).reshape(1, -1)

    # Validate
    assert x.shape == (1, 6), f"Feature shape error: {x.shape}"
    return x


# ── M1 → M3 Feature Proxy Bridge ─────────────────────────────────────────────
def map_m1_to_m3_features(
    m1_features: Dict[str, float],
) -> Dict[str, float]:
    """
    Map M1 CNN AR features (f0–f11, normalized) to M3 SHARP features (f0–f5).

    ⚠️  APPROXIMATION — For academic demonstration only when live JSOC data
    is unavailable. M1's CNN features are dimensionless; they are used as
    ORDER-OF-MAGNITUDE proxies for the SHARP parameters.

    The mapping is documented in M1_TO_M3_MAP and M1_TO_M3_WEIGHTS.
    When SHARP data IS available, M3 uses it directly (not this function).

    Args:
        m1_features: Dict with keys f0..f11 (M1 output contract)

    Returns:
        Dict with keys f0..f5 in approximate physical units
    """
    # Reference scale factors to convert M1 normalized [0,3] range
    # to approximate SHARP physical units (geometric mean of typical values)
    SCALE_FACTORS = {
        "f0": 1e22,   # TOTUSJH typical: 1e20–1e24
        "f1": 1e25,   # TOTPOT typical: 1e23–1e27
        "f2": 1e13,   # TOTUSJZ typical: 1e11–1e15
        "f3": 1e15,   # ABSNJZH typical: 1e13–1e17
        "f4": 5e3,    # R_VALUE typical: 1e2–1e5
        "f5": 5e2,    # AREA_ACR typical: 1e1–1e4
    }

    result = {}
    for m3_feat, m1_sources in M1_TO_M3_MAP.items():
        weights = M1_TO_M3_WEIGHTS[m3_feat]
        weighted_sum = sum(
            w * float(m1_features.get(src, 0.0))
            for w, src in zip(weights, m1_sources)
        )
        # Scale to approximate physical units
        result[m3_feat] = weighted_sum * SCALE_FACTORS[m3_feat]

    logger.debug(
        "[M3] M1→M3 proxy mapping applied. "
        "Results are approximate — use SHARP data for production."
    )
    return result


# ── Feature summary ───────────────────────────────────────────────────────────
def get_feature_names() -> List[str]:
    """Return the ordered list of M3 feature column names."""
    return FEATURE_COLS.copy()


def print_feature_summary() -> None:
    """Print a human-readable summary of M3 features."""
    print("\nM3 Sentinel — SHARP Feature Summary")
    print("=" * 55)
    for feat in FEATURE_COLS:
        desc = FEATURE_DESCRIPTIONS[feat]
        lo, hi = FEATURE_CLIP_BOUNDS[feat]
        print(f"  {feat}: {desc}")
        print(f"       Clip range: [{lo:.0e}, {hi:.0e}]  Transform: log1p")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_feature_summary()

    try:
        X, y, names = build_feature_matrix()
        print(f"Feature matrix: {X.shape}")
        print(f"Label balance: {y.mean()*100:.2f}% positive")
        print(f"Feature ranges (after log1p):")
        for i, name in enumerate(names):
            print(f"  {name}: min={X[:,i].min():.3f}, max={X[:,i].max():.3f}, "
                  f"mean={X[:,i].mean():.3f}")
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nTest with M1 proxy mapping instead:")
        m1_mock = {f"f{i}": float(i * 0.3 + 0.1) for i in range(12)}
        m3_feats = map_m1_to_m3_features(m1_mock)
        print("M1→M3 mapped features:")
        for k, v in m3_feats.items():
            print(f"  {k}: {v:.3e}")
