"""
ICARUS-X — M3 Sentinel: Data Downloader (INDEPENDENT MODULE)

Downloads SHARP solar AR parameters and GOES flare labels for M3 training.
M3 is fully independent of M1, M2, and M4 — this file is its sole data source.

Data Sources:
  Primary   (Option A): JSOC DRMS API → hmi.sharp_cea_720s series
  Secondary (Option B): SWAN-SF dataset from Zenodo (Angryk et al. 2020)
  Fallback  (Option C): Synthetic SHARP-like data (testing/CI only)

Output:
  data/sharp_flare_dataset.csv
    Columns: timestamp, NOAA_AR, label (0/1), f0–f5
    label=1 → M-class or X-class flare within 24h
    label=0 → no flare or C-class

Usage:
  python m3_classifier/data_download.py --source zenodo
  python m3_classifier/data_download.py --source jsoc
  python m3_classifier/data_download.py --source synthetic
"""

import argparse
import sys
import time
import zipfile
import io
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
from loguru import logger

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_DIR / "sharp_flare_dataset.csv"

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    # SWAN-SF dataset (Angryk et al. 2020) — pre-computed SHARP + labels
    # DOI: 10.5281/zenodo.4603199
    "zenodo_url": "https://zenodo.org/record/4603199/files/SWAN-SF.zip?download=1",
    "zenodo_timeout_s": 300,

    # JSOC DRMS API base URL
    "jsoc_base_url": "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info",
    "jsoc_email": "icarus.x.project@gmail.com",  # register at jsoc.stanford.edu

    # GOES event catalog (NOAA NGDC)
    "goes_url": (
        "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
        "solar-features/solar-flares/x-rays/goes/xrs/"
        "goes-xrs-report_2010-2023_lates.txt"
    ),

    # Synthetic params
    "synthetic_n_samples": 20_000,
    "synthetic_positive_rate": 0.03,   # ~3% M/X flare rate (realistic)
    "random_seed": 42,
}

# ── SHARP feature mapping (SWAN-SF column → our feature name) ────────────────
SHARP_FEATURE_MAP = {
    "TOTUSJH":  "f0",   # total unsigned current helicity
    "TOTPOT":   "f1",   # total photospheric magnetic free energy proxy
    "TOTUSJZ":  "f2",   # total unsigned vertical current
    "ABSNJZH":  "f3",   # absolute value of net current helicity
    "R_VALUE":  "f4",   # flux near polarity inversion line
    "AREA_ACR": "f5",   # area of strong-field pixels in active region
}

# Fallback if SWAN-SF uses slightly different column names
SHARP_FEATURE_ALIASES = {
    "TOTUSJH": ["TOTUSJH", "totusjh", "total_us_jh"],
    "TOTPOT":  ["TOTPOT",  "totpot",  "total_pot"],
    "TOTUSJZ": ["TOTUSJZ", "totusjz", "total_us_jz"],
    "ABSNJZH": ["ABSNJZH", "absnjzh", "abs_njzh", "TOTBSQ", "totbsq"],
    "R_VALUE": ["R_VALUE", "r_value", "rvalue"],
    "AREA_ACR":["AREA_ACR","area_acr","area"],
}

FEATURE_DESCRIPTIONS = {
    "f0": "total_unsigned_current_helicity (TOTUSJH)",
    "f1": "free_energy_proxy (TOTPOT)",
    "f2": "total_unsigned_vertical_current (TOTUSJZ)",
    "f3": "lorentz_force_proxy (ABSNJZH)",
    "f4": "pil_flux (R_VALUE)",
    "f5": "field_area (AREA_ACR)",
}


# ── Option B: SWAN-SF / Zenodo Download ───────────────────────────────────────
def download_zenodo(url: str = CONFIG["zenodo_url"]) -> pd.DataFrame:
    """
    Download SWAN-SF dataset from Zenodo and extract SHARP features + labels.

    Args:
        url: Zenodo zip URL

    Returns:
        DataFrame with columns [timestamp, NOAA_AR, label, f0..f5]
    """
    logger.info(f"[ZENODO] Downloading SWAN-SF dataset from:\n  {url}")
    logger.info("  This may take a few minutes (file ~200 MB)...")

    try:
        resp = requests.get(url, timeout=CONFIG["zenodo_timeout_s"], stream=True)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunks = []
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            chunks.append(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                logger.info(f"  Progress: {downloaded/1e6:.1f} / {total/1e6:.1f} MB ({pct:.0f}%)")

        raw = b"".join(chunks)
        logger.info("[ZENODO] Download complete. Extracting...")

        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            logger.info(f"  Found {len(csv_files)} CSV files: {csv_files[:5]}")

            dfs = []
            for fname in csv_files:
                try:
                    with zf.open(fname) as f:
                        df = pd.read_csv(f, low_memory=False)
                    dfs.append(df)
                    logger.info(f"  Loaded {fname}: {df.shape}")
                except Exception as e:
                    logger.warning(f"  Skipping {fname}: {e}")

        if not dfs:
            raise ValueError("No valid CSV files found in SWAN-SF archive.")

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"[ZENODO] Combined shape: {combined.shape}")

        return _parse_swan_sf(combined)

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not reach Zenodo. Check your internet connection.\n"
            "Alternative: python m3_classifier/data_download.py --source synthetic"
        )


def _parse_swan_sf(df: pd.DataFrame) -> pd.DataFrame:
    """Parse SWAN-SF columns into M3 standard format."""
    logger.info("[PARSE] Mapping SWAN-SF columns to M3 feature format...")

    out = pd.DataFrame()

    # Timestamp
    for ts_col in ["T_REC", "timestamp", "time", "DATE", "date"]:
        if ts_col in df.columns:
            out["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
            logger.info(f"  Timestamp column: '{ts_col}'")
            break
    else:
        logger.warning("  No timestamp column found — using row index as time")
        out["timestamp"] = pd.date_range("2010-01-01", periods=len(df), freq="12min")

    # NOAA AR number
    for ar_col in ["NOAA_AR", "harpnum", "HARPNUM", "AR_NUM"]:
        if ar_col in df.columns:
            out["NOAA_AR"] = df[ar_col].fillna(0).astype(int)
            break
    else:
        out["NOAA_AR"] = 0

    # Label: flare class ≥ M → 1
    label_assigned = False
    for lbl_col in ["label", "LABEL", "flare_class", "CLASS", "goes_class"]:
        if lbl_col in df.columns:
            raw_labels = df[lbl_col].astype(str).str.upper().str.strip()
            # M or X class → positive
            out["label"] = raw_labels.apply(
                lambda c: 1 if (c.startswith("M") or c.startswith("X")) else 0
            )
            label_assigned = True
            pos = out["label"].sum()
            logger.info(f"  Label column: '{lbl_col}' | Positives: {pos} / {len(out)} ({pos/len(out)*100:.1f}%)")
            break

    if not label_assigned:
        logger.warning("  No label column found — all labels set to 0 (recheck dataset)")
        out["label"] = 0

    # SHARP features
    for sharp_key, feat_name in SHARP_FEATURE_MAP.items():
        resolved_col = None
        for alias in SHARP_FEATURE_ALIASES.get(sharp_key, [sharp_key]):
            if alias in df.columns:
                resolved_col = alias
                break

        if resolved_col:
            vals = pd.to_numeric(df[resolved_col], errors="coerce")
            # Replace fill values (JSOC uses large sentinels like -1e31)
            vals = vals.where(vals > -1e20, np.nan)
            out[feat_name] = vals
            logger.info(f"  {feat_name} ← '{resolved_col}': {vals.notna().sum()} valid values")
        else:
            logger.warning(f"  {feat_name} ({sharp_key}): column not found — filled with 0")
            out[feat_name] = 0.0

    # Drop rows with NaN in ANY feature
    before = len(out)
    out = out.dropna(subset=[f"f{i}" for i in range(6)])
    logger.info(f"[PARSE] Dropped {before - len(out)} rows with NaN features. Final: {len(out)} rows")

    return out


# ── Option A: JSOC DRMS API ───────────────────────────────────────────────────
def download_jsoc(
    email: str = CONFIG["jsoc_email"],
    start: str = "2010.01.01_00:00:00_TAI",
    stop: str = "2023.12.31_23:59:00_TAI",
) -> pd.DataFrame:
    """
    Download SHARP parameters directly from JSOC.
    Requires drms package and a registered JSOC email.

    Args:
        email: Registered JSOC email address
        start: Start time in JSOC TAI format
        stop: End time in JSOC TAI format

    Returns:
        DataFrame with M3 standard columns
    """
    try:
        import drms
    except ImportError:
        raise ImportError(
            "drms package not installed. Run: pip install drms\n"
            "Then register your email at: http://jsoc.stanford.edu/ajax/register_email.html"
        )

    logger.info(f"[JSOC] Connecting to JSOC DRMS (email={email})...")
    client = drms.Client(email=email, verbose=True)

    query = f"hmi.sharp_cea_720s[][{start}-{stop}@12h]"
    segments = list(SHARP_FEATURE_MAP.keys()) + ["T_REC", "NOAA_AR"]

    logger.info(f"[JSOC] Query: {query}")
    logger.info("  This may take 10–30 minutes. JSOC rate-limits requests.")

    try:
        keys = client.query(query, key=",".join(segments))
        logger.info(f"[JSOC] Retrieved {len(keys)} records")
        return _parse_swan_sf(keys)
    except Exception as e:
        raise RuntimeError(f"JSOC query failed: {e}")


# ── Option C: Synthetic Fallback ──────────────────────────────────────────────
def generate_synthetic(
    n_samples: int = CONFIG["synthetic_n_samples"],
    positive_rate: float = CONFIG["synthetic_positive_rate"],
    seed: int = CONFIG["random_seed"],
) -> pd.DataFrame:
    """
    Generate synthetic SHARP-like data for CI/testing.

    ⚠️  WARNING: This is NOT real solar data. Do NOT use for published results.
    For academic demonstration only when JSOC/Zenodo is unavailable.

    Args:
        n_samples: Number of AR parameter snapshots
        positive_rate: Fraction of M/X flare events (~0.02–0.05 is realistic)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with M3 standard columns
    """
    logger.warning("=" * 60)
    logger.warning("[!] SYNTHETIC MODE — NOT real SHARP data")
    logger.warning("    Use --source zenodo or --source jsoc for real training")
    logger.warning("=" * 60)

    rng = np.random.RandomState(seed)
    timestamps = pd.date_range("2010-01-01", periods=n_samples, freq="12min")

    # Realistic SHARP value ranges (log-normal, physics-informed)
    # Reference: Bobra & Couvidat 2015, Table 1 value ranges
    data = {
        "timestamp": timestamps,
        "NOAA_AR": rng.randint(11000, 13000, n_samples),
        "label": (rng.random(n_samples) < positive_rate).astype(int),
    }

    # Quiet-sun baseline SHARP values (log-normal)
    sharp_params = {
        "f0": (5e21, 2.0),   # TOTUSJH  A/m²  log-mean, log-std
        "f1": (1e24, 1.5),   # TOTPOT   erg/cm³
        "f2": (5e12, 2.0),   # TOTUSJZ  A
        "f3": (1e14, 1.5),   # ABSNJZH  G²/m
        "f4": (5e2,  1.8),   # R_VALUE  Mx
        "f5": (1e3,  1.5),   # AREA_ACR μHem
    }

    for feat, (log_mean, log_std) in sharp_params.items():
        quiet = rng.lognormal(np.log(log_mean), log_std, n_samples)
        # Flare events: features elevated by 2–5× (physically motivated)
        flare_boost = np.where(data["label"] == 1, rng.uniform(2.0, 5.0, n_samples), 1.0)
        data[feat] = quiet * flare_boost

    df = pd.DataFrame(data)
    pos = df["label"].sum()
    logger.info(f"[SYNTHETIC] Generated {n_samples} samples | Positives: {pos} ({pos/n_samples*100:.1f}%)")
    return df


# ── Save / Validate ───────────────────────────────────────────────────────────
def validate_and_save(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """
    Validate the dataset and save to CSV.

    Args:
        df: Processed DataFrame with required columns
        path: Output path
    """
    logger.info("[VALIDATE] Checking dataset quality...")

    required_cols = ["timestamp", "label"] + [f"f{i}" for i in range(6)]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Feature validation
    for feat in [f"f{i}" for i in range(6)]:
        n_zero = (df[feat] == 0).sum()
        pct_zero = n_zero / len(df) * 100
        if pct_zero > 50:
            logger.warning(
                f"  [!] {feat}: {pct_zero:.0f}% zero values — "
                f"{FEATURE_DESCRIPTIONS[feat]} may not have loaded correctly"
            )

    # Label balance
    pos = df["label"].sum()
    neg = len(df) - pos
    if pos == 0:
        raise ValueError(
            "No positive (M/X flare) labels found. "
            "Check your data source — the flare catalog may not have loaded."
        )
    logger.info(f"  Label balance: {pos} positive / {neg} negative ({pos/len(df)*100:.2f}% positive)")

    # Minimum size
    if len(df) < 1000:
        logger.warning(f"  [!] Only {len(df)} samples — recommend ≥10,000 for reliable training")

    # Save
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"[SAVE] Dataset saved to: {path}")
    logger.info(f"       Shape: {df.shape}")
    logger.info(f"       Columns: {list(df.columns)}")

    # Print feature summary
    logger.info("\n[SUMMARY] Feature statistics:")
    for feat in [f"f{i}" for i in range(6)]:
        desc = FEATURE_DESCRIPTIONS[feat]
        mean = df[feat].mean()
        std = df[feat].std()
        logger.info(f"  {feat} ({desc[:30]:30s}): mean={mean:.3e}, std={std:.3e}")


# ── Entry Point ───────────────────────────────────────────────────────────────
def main(source: str = "zenodo") -> None:
    """
    Main download entry point.

    Args:
        source: 'zenodo' | 'jsoc' | 'synthetic'
    """
    logger.info("=" * 60)
    logger.info("ICARUS-X M3 Sentinel — Data Download")
    logger.info(f"Source: {source.upper()}")
    logger.info("=" * 60)

    start_time = time.time()

    if source == "zenodo":
        df = download_zenodo()
    elif source == "jsoc":
        df = download_jsoc()
    elif source == "synthetic":
        df = generate_synthetic()
    else:
        raise ValueError(f"Unknown source '{source}'. Choose: zenodo | jsoc | synthetic")

    validate_and_save(df)

    elapsed = time.time() - start_time
    logger.info(f"\n[DONE] Completed in {elapsed:.1f}s")
    logger.info(f"       Next step: python m3_classifier/train_xgb.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ICARUS-X M3: Download SHARP flare training data"
    )
    parser.add_argument(
        "--source",
        choices=["zenodo", "jsoc", "synthetic"],
        default="zenodo",
        help=(
            "zenodo = SWAN-SF dataset (recommended for deadlines); "
            "jsoc   = raw SHARP from JSOC DRMS (most accurate); "
            "synthetic = mock data for CI/testing only"
        ),
    )
    args = parser.parse_args()
    main(args.source)
