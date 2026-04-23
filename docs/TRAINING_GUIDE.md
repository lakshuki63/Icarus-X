# ICARUS-X — Training Guide

> **Version:** 1.0 | All 4 models trained independently, in order.

---

## Prerequisites

```bash
pip install -r requirements.txt
# Verify key packages
python -c "import xgboost, torch, optuna, shap; print('OK')"
```

---

## Step 1 — Download OMNI Solar Wind Data (M2)

```bash
# Option A: Use the built-in NOAA downloader
python m2_predictor/data_loader.py

# Option B: Manual download from OMNIWeb
# https://omniweb.gsfc.nasa.gov/form/dx1.html
# Select: 1-min resolution, 2010–2023
# Variables: Bx, By, Bz (GSM), |B|, Vsw, Np, T
# Save as: data/omni_solar_wind.csv

# Option C: Use the Sept 2017 storm data (demo only)
# Already included at: data/omni_solar_wind_demo.csv
```

---

## Step 2 — Train M2 BiGRU (Kp Predictor)

```bash
# Default (50 epochs, CPU)
python m2_predictor/train.py

# With GPU and more epochs
python m2_predictor/train.py --epochs 80 --device cuda

# Verify output
ls models/bigru_predictor.pt      # checkpoint
ls models/feature_scaler.pkl      # scaler

# Evaluate (per-horizon RMSE vs persistence baseline)
python m2_predictor/evaluate.py
```

**Expected output:**
```
Horizon | Model RMSE | Persist RMSE | vs Persist
  +3h   |    0.91    |     1.24     |    -26.6% ✅
  +6h   |    1.03    |     1.41     |    -26.9% ✅
  ...
```

> **Note:** If AR features (f0..f11) are all-zero, M2 trains with SW only (7 features).  
> The checkpoint is always saved with `n_features=19` (padded) for infer.py compat.

---

## Step 3 — Download SHARP Flare Data (M3)

M3 is **fully independent** — do NOT use solar wind data for this step.

```bash
# Option A: SWAN-SF dataset from Zenodo (~200 MB, fastest)
python m3_classifier/data_download.py --source zenodo

# Option B: Raw SHARP from JSOC DRMS (most accurate, 10–30 min)
# 1. Register at: http://jsoc.stanford.edu/ajax/register_email.html
# 2. Set email in m3_classifier/data_download.py CONFIG["jsoc_email"]
python m3_classifier/data_download.py --source jsoc

# Option C: Synthetic data (testing only — NOT for published results)
python m3_classifier/data_download.py --source synthetic

# Verify output
python -c "import pandas as pd; df=pd.read_csv('data/sharp_flare_dataset.csv'); print(df.shape, df['label'].mean())"
```

**Expected output:** `(N_rows, 8)   0.025`  (2–5% positive flare rate is realistic)

---

## Step 4 — Train M3 XGBoost (Flare Classifier)

```bash
# Default (50 Optuna trials)
python m3_classifier/train_xgb.py

# More trials for better HPO (recommended if time allows)
python m3_classifier/train_xgb.py --trials 100

# Verify output
ls models/xgb_flare_sentinel.json
ls models/xgb_flare_sentinel_meta.json

# Evaluate (TSS + F2)
python m3_classifier/evaluate.py
```

**Expected output:**
```
TSS  (target >0.65): 0.71  ✅
F2   (target >0.70): 0.74  ✅
ROC-AUC:             0.87
```

> **If metrics below target:** Try `--source jsoc` for real SHARP data, then retrain.

---

## Step 5 — M4 GIC (No training required)

M4 uses the empirical formula `GIC = 10^(0.28×Kp − 0.90)` from Pulkkinen et al. (2012).  
No training needed. Verify the formula:

```bash
python m4_gic/gic_model.py
# Expected: R² > 0.75 vs Ngwira 2015 reference table
```

If FINGES observational data is available:
```python
from m4_gic.gic_model import fit_from_data
import numpy as np
kp_obs  = np.array([...])   # observed Kp values
gic_obs = np.array([...])   # corresponding GIC measurements
result  = fit_from_data(kp_obs, gic_obs)
print(result)  # {'a': ..., 'b': ..., 'r_squared': ...}
```

---

## Step 6 — Run the Full System

```bash
# Start the FastAPI server
uvicorn m5_architect.main:app --host 0.0.0.0 --port 8000 --reload

# Open browser
http://localhost:8000
```

---

## Training Order Summary

| Step | Module | Script | Time | Checkpoint |
|------|--------|--------|------|------------|
| 1 | Data | `m2_predictor/data_loader.py` | ~5 min | `data/omni_solar_wind.csv` |
| 2 | M2 | `m2_predictor/train.py` | ~20–40 min | `models/bigru_predictor.pt` |
| 3 | Data | `m3_classifier/data_download.py` | ~10 min | `data/sharp_flare_dataset.csv` |
| 4 | M3 | `m3_classifier/train_xgb.py` | ~15–30 min | `models/xgb_flare_sentinel.json` |
| 5 | M4 | *(no training)* | — | — |
| 6 | Run | `uvicorn m5_architect.main:app` | — | — |

---

## Troubleshooting

### M2: `n_features mismatch` on checkpoint load
```bash
# The checkpoint was saved with n_features=7 (AR missing during training)
# Fix: retrain with AR features
python m1_visionary/export_features.py  # generate ar_features.csv
python m2_predictor/train.py            # retrain (will now use 19 features)
```

### M3: `F2 < 0.50` after training
```bash
# 1. Check positive rate in dataset
python -c "import pandas as pd; df=pd.read_csv('data/sharp_flare_dataset.csv'); print(df['label'].mean())"
# If < 0.01 → dataset too small or wrong labels → re-download with --source jsoc

# 2. Try higher Optuna budget
python m3_classifier/train_xgb.py --trials 100

# 3. Check optimal threshold
python m3_classifier/evaluate.py --threshold 0.3
```

### `optuna not found`
```bash
pip install optuna>=3.6.0
```

### Windows DataLoader error
DataLoaders are set to `num_workers=0` by default to avoid Windows multiprocessing errors. This is correct and expected.
