# ICARUS-X — System Architecture

> **Version:** 1.0 | **Last Updated:** 2026-04-23

---

## Overview

ICARUS-X is a real-time space weather forecasting system that chains four ML modules (M1–M4) orchestrated by M5, delivering Kp forecasts and GIC risk assessments via a FastAPI/WebSocket server to a browser dashboard.

---

## Module Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                                    │
│   DSCOVR/ACE   NOAA Kp   GOES-16   JSOC SHARP   GOES Flare Cat.   │
└────┬──────────────┬──────────────────┬──────────────┬──────────────┘
     │              │                  │              │
     ▼              ▼                  ▼              ▼
 Solar Wind      Kp Index         M1 VISIONARY    M3 SHARP DATA
 (OMNI CSV)     (3-hr avg)       (YOLOv10 CNN)   (ar_features.csv)
                                  ▼ f0..f11         ▼ f0..f5 (SHARP)
                                  └────────┬─────────┘
                                           │
          ┌────────────────────────────────┼────────────────────────────┐
          │                               │                            │
          ▼                               ▼                            │
   ┌──────────────┐              ┌──────────────────┐                  │
   │  M2 PREDICTOR│              │  M3 SENTINEL      │  ← INDEPENDENT  │
   │  BiGRU Seq2Seq│             │  XGBoost Binary   │                  │
   │  (SW + AR in)│              │  Flare Classifier │                  │
   │  Kp ×8 out  │              │  (SHARP in)       │                  │
   └──────┬───────┘              └────────┬──────────┘                  │
          │                               │                            │
          ▼                               │                            │
   ┌──────────────┐                       │                            │
   │   M4 GIC     │                       │                            │
   │  Empirical   │                       │                            │
   │  (Kp → GIC)  │                       │                            │
   └──────┬───────┘                       │                            │
          │                               │                            │
          └───────────────┬───────────────┘                            │
                          ▼                                            │
                   ┌──────────────┐                                    │
                   │  M5 ARCHITECT│ ◄──────────────────────────────────┘
                   │  Orchestrator│
                   │  (Merger)    │
                   └──────┬───────┘
                          ▼
                   WebSocket Payload:
                   {timestamp, kp_forecast,
                    flare, gic_risk,
                    solar_wind_latest,
                    data_quality}
                          ▼
                   FastAPI /ws/live
                          ▼
                   Browser Dashboard
```

> **Critical Design Rule:** M3 Sentinel is **fully independent** of M1, M2, and M4.  
> M3 reads SHARP parameters (not solar wind data). M5 merges all outputs — not M4.

---

## Module Contracts

### M1 Visionary (YOLOv10 CNN)
- **Input:** Magnetogram image (512×512, grayscale)
- **Output:** `ar_features.csv` → `f0..f11` (12 AR CNN features, normalized 0–3)
- **Checkpoint:** `models/yolov10/best.pt`
- **Fallback:** Zero vector `[0.0] × 12` (not random stub)

### M2 Predictor (BiGRU Seq2Seq)
- **Input:** 24h × 19 features (7 SW columns + 12 AR features)
- **Output:** 8 Kp forecasts at +3h, +6h, ..., +24h with MC uncertainty
- **Checkpoint:** `models/bigru_predictor.pt`
- **Attention:** 24-float array summing to 1.0

### M3 Sentinel (XGBoost Binary)
- **Input:** 6 SHARP parameters → `f0..f5` (log1p transformed)
- **Output:** `{flare_probability, flare_class, predicted_tier, confidence, top_features}`
- **Checkpoint:** `models/xgb_flare_sentinel.json`
- **Labels:** 0 = no/C-class flare, 1 = M/X-class flare
- **Target:** TSS > 0.65, F2 > 0.70

### M4 GIC (Empirical Formula)
- **Input:** M2 Kp forecast (8 horizons)
- **Output:** 8 GIC risk horizons + headline alert
- **Formula:** `GIC = 10^(0.28×Kp - 0.90)` [Pulkkinen 2012]
- **Risk tiers:** LOW (<1 A/km) | MEDIUM (<5) | HIGH (<15) | CRITICAL (≥15)

### M5 Architect (Orchestrator)
- **Input:** `solar_wind_readings` (list of dicts from DB poller)
- **Output:** 6-key merged dict (WebSocket payload)
- **Merge logic:** M5 merges M2+M3+M4; M4 is NOT passed M3 output

---

## Output Contract (WebSocket)

```json
{
  "timestamp": "2026-04-23T15:44:00Z",
  "kp_forecast": {
    "run_timestamp": "...",
    "horizons": [
      {
        "horizon_hr": 3,
        "kp_predicted": 4.2,
        "kp_ci_low": 3.1,
        "kp_ci_high": 5.3,
        "kp_std": 0.6,
        "attention_weights": [0.04, 0.04, ...]
      }
    ],
    "is_stub": false
  },
  "flare": {
    "flare_probability": 0.83,
    "flare_class": "X",
    "predicted_tier": "X-class",
    "confidence": "HIGH",
    "top_features": [{"name": "free_energy_proxy (TOTPOT)", "shap_value": 0.51}],
    "is_stub": false
  },
  "gic_risk": {
    "run_timestamp": "...",
    "horizons": [
      {
        "horizon_hr": 3,
        "kp_predicted": 4.2,
        "gic_mean": 1.66,
        "gic_p5": 0.8,
        "gic_p95": 3.4,
        "risk_tier": "MEDIUM",
        "risk_tier_worst": "HIGH",
        "risk_color": "#FFA500"
      }
    ],
    "headline_alert": {
      "alert_level": "HIGH",
      "peak_horizon_hr": 9,
      "peak_gic_estimate": 8.4,
      "message": "...",
      "should_email": true,
      "should_sms": false
    },
    "is_stub": false
  },
  "solar_wind_latest": {"bz": -28.0, "vsw": 770.0, "np": 15.0, "pdyn": 4.2},
  "data_quality": {
    "m1_real": false,
    "m2_real": true,
    "m3_real": true,
    "m4_real": true
  }
}
```

---

## FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Serve `frontend/index.html` |
| `WS`   | `/ws/live` | WebSocket — pushes forecast every 60s |
| `GET`  | `/api/forecast/latest` | Latest forecast (REST fallback for WS) |
| `GET`  | `/api/solar-wind/latest` | Latest DSCOVR reading |
| `GET`  | `/health` | System health + data_quality flags |

---

## Directory Structure

```
Icarus-X/
├── m1_visionary/       # YOLOv10 AR feature extractor
├── m2_predictor/       # BiGRU Kp forecast (SW + AR)
├── m3_classifier/      # XGBoost flare classifier (SHARP, INDEPENDENT)
├── m4_gic/             # Empirical GIC risk model
├── m5_architect/       # Orchestrator + FastAPI server
├── frontend/           # React dashboard (index.html)
├── data/               # CSVs (omni, kp, ar_features, sharp_flare_dataset)
├── models/             # Checkpoints (.pt, .json, .pkl)
├── docs/               # This directory
└── requirements.txt
```
