# ICARUS-X — Deployment & Demo Guide

> **Version:** 1.0 | September 2017 Storm Replay Demo included.

---

## Quick Start (Development)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Train models — skip to use fallback mode
python m2_predictor/train.py
python m3_classifier/data_download.py --source donki
python m3_classifier/train_xgb.py

# 3. Start server
uvicorn m5_architect.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Open dashboard
http://localhost:8000
```

---

## Demo Mode (No trained models required)

The system starts in **fallback mode** when checkpoints are missing:
- M2 → Kp=0 at all horizons, `m2_real=false`
- M3 → `flare_probability=0.0`, `m3_real=false`
- M4 → LOW risk, `is_stub=true`
- Dashboard shows yellow `~proxy` badges per module

The system is fully functional — the WebSocket still pushes every 60s.

---

## September 2017 Storm Replay Demo

The September 2017 X9.3 solar flare event (2017-09-06) is the reference storm for ICARUS-X validation.

### Using synthetic data (instant, no download)

```bash
# Step 1: Generate synthetic data centered on Sept 2017 storm profile
python m3_classifier/data_download.py --source synthetic

# Step 2: Train M3 on synthetic data
python m3_classifier/train_xgb.py --trials 20

# Step 3: Start server
uvicorn m5_architect.main:app --port 8000

# Step 4: Replay Sept 2017 via REST
curl -X POST http://localhost:8000/api/debug/replay \
  -H "Content-Type: application/json" \
  -d '{"event": "sept_2017"}'
```

### Using real OMNI data (recommended for presentations)

```bash
# Download Sept 2017 OMNI data from OMNIWeb
# https://omniweb.gsfc.nasa.gov/form/dx1.html
# Date range: 2017-09-01 to 2017-09-10
# Save as: data/omni_sept2017.csv

# Replay via API
curl -X POST http://localhost:8000/api/debug/replay \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "data/omni_sept2017.csv"}'
```

**Expected dashboard behavior during replay:**
- Bz drops to -28 nT at ~12:00 UTC Sept 6
- Kp forecast rises to 8–9 within 3h horizon
- GIC risk escalates to CRITICAL (~18 A/km)
- M3 flare panel: probability > 0.80, class = X
- Alert banner turns RED with `CRITICAL` badge

---

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV USE_REAL_M1=true

CMD ["uvicorn", "m5_architect.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t icarus-x .
docker run -p 8000:8000 -v $(pwd)/models:/app/models icarus-x
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_REAL_M1` | `false` | Enable real YOLOv10 magnetogram inference |
| `DB_URL` | `sqlite:///icarus.db` | Database connection string |
| `WS_PUSH_INTERVAL` | `60` | WebSocket push interval (seconds) |
| `NOAA_POLL_INTERVAL` | `60` | DSCOVR/ACE polling interval (seconds) |
| `LOG_LEVEL` | `INFO` | Loguru log level |

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "data_quality": {
    "m1_real": false,
    "m2_real": true,
    "m3_real": true,
    "m4_real": true
  },
  "last_forecast": "2026-04-23T15:44:00Z",
  "solar_wind_age_s": 42
}
```

---

## Data Sources

| Source | URL | Module | Update Freq |
|--------|-----|--------|-------------|
| DSCOVR/ACE solar wind | `https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json` | M2 | 1 min |
| NOAA Kp index | `https://services.swpc.noaa.gov/json/planetary_k_index_1m.json` | M2 | 1 min |
| GOES X-ray flux | `https://services.swpc.noaa.gov/json/goes/primary/xrays-1-minute.json` | M3 alert | 1 min |
| JSOC SHARP | `http://jsoc.stanford.edu` | M3 training | 12 min cadence |
| NASA DONKI | `https://api.nasa.gov/` | M3 training | Live API |

---

## Known Limitations

1. **M1 Visionary**: YOLOv10 magnetogram model requires `models/yolov10/best.pt` and Kaggle training data (AR bounding boxes). For demo, use zero-vector fallback (`m1_real=false`).

2. **M3 in proxy mode**: When M1 is not real, M3 uses `map_m1_to_m3_features()` which is an approximation (documented in `m3_classifier/features.py`). This is acceptable for demo; use JSOC SHARP data for production accuracy.

3. **GIC formula**: Parameters `a=0.28, b=-0.90` from Pulkkinen (2012) are not fitted to local FINGES data. Call `m4_gic.gic_model.fit_from_data()` if local GIC measurements are available.

4. **WebSocket reconnection**: Frontend implements exponential backoff (1s → 2s → 4s → 8s, max 30s). If server is restarted, browser reconnects automatically within 30s.

---

## Academic Citation

If using ICARUS-X for research, please cite:

```
ICARUS-X Space Weather Forecasting System
Training data: NASA DONKI FLR API
GIC model: Pulkkinen et al. (2012) Space Weather, 10, S08009
Feature selection: Bobra & Couvidat (2015) ApJ, 798, 135
```
