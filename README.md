# ICARUS-X 🛰️

**Space Weather Forecasting & GIC Risk Assessment System**

Real-time prediction of geomagnetic storms and geomagnetically induced current (GIC) risk using multi-source data fusion from NASA SDO magnetograms and NOAA DSCOVR solar wind measurements.

---

## 🏗️ Architecture

```
SDO Magnetogram ──→ M1 Visionary (YOLOv10) ──┐
                                               ├──→ M2 Predictor (BiGRU) ──→ M3 Sentinel (XGBoost) ──→ M4 GIC Risk ──→ M5 Dashboard
NOAA DSCOVR ──────→ Real-time Solar Wind ─────┘
```

| Module | Purpose | Model |
|--------|---------|-------|
| **M1 Visionary** | Detect active regions in magnetograms | YOLOv10 + CNN Feature Head |
| **M2 Predictor** | Forecast Kp index (3-24h ahead) | Bidirectional GRU + Attention |
| **M3 Sentinel** | Classify storm severity (G0-G4) | XGBoost + Focal Loss |
| **M4 GIC Risk** | Estimate GIC current risk | Empirical model + MC Dropout |
| **M5 Architect** | API + Dashboard | FastAPI + React |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- (Optional) PostgreSQL 15+ — falls back to SQLite automatically

### Setup

```bash
# 1. Clone and enter project
cd icarus-x

# 2. Create .env from template
copy .env.template .env

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the demo test (verifies all modules)
python demo_test.py

# 6. Start the server
python -m uvicorn m5_architect.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Open dashboard (in another terminal)
python -m http.server 3000 --directory frontend

# 8. Open browser
# Dashboard: http://localhost:3000
# API Docs:  http://localhost:8000/docs
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/status` | System status + model availability |
| GET | `/api/forecast/latest` | Latest forecast result |
| GET | `/api/forecast/history` | Recent forecast runs |
| POST | `/api/forecast/run` | Trigger manual forecast |
| GET | `/api/solar-wind/latest` | Latest NOAA reading |
| GET | `/api/solar-wind/history` | Recent solar wind data |
| GET | `/api/alerts` | Recent alerts |
| WS | `/ws/live` | Real-time forecast push (60s) |

---

## 🔌 Plugging In Trained Models

When your YOLOv10 training on Kaggle finishes:

```bash
# 1. Download from Kaggle
#    /kaggle/working/models/yolov10_magnetogram/weights/best.pt
#    /kaggle/working/models/feature_head_best.pt

# 2. Copy to project
mkdir -p models/yolov10
cp best.pt models/yolov10/best.pt
cp feature_head_best.pt models/feature_head_best.pt

# 3. Update .env
#    USE_REAL_M1=true

# 4. Restart server — it auto-detects the checkpoints
```

### Training M2 (BiGRU Predictor)
```bash
python -m m2_predictor.train
# Checkpoint saved to models/bigru_predictor.pt
```

### Training M3 (XGBoost Sentinel)
```bash
python -m m3_classifier.train_xgb
# Checkpoint saved to models/xgb_sentinel.json
```

---

## 📁 Project Structure

```
icarus-x/
├── m1_visionary/          # YOLOv10 active region detection
│   ├── visionary.py       # Real YOLO pipeline
│   ├── visionary_stub.py  # Mock AR features
│   └── feature_extractor.py  # CNN feature head
├── m2_predictor/          # Kp index forecasting
│   ├── model.py           # BiGRU + Bahdanau Attention
│   ├── train.py           # Training loop
│   ├── infer.py           # Inference + stub
│   ├── data_loader.py     # OMNI data loading
│   └── windowing.py       # Sliding window dataset
├── m3_classifier/         # Storm severity classification
│   ├── train_xgb.py       # XGBoost + SMOTE training
│   ├── infer.py           # Classification + stub
│   ├── features.py        # Feature engineering
│   └── storm_events.py    # G-scale mapping
├── m4_gic/                # GIC risk assessment
│   ├── gic_model.py       # Empirical Kp→GIC model
│   ├── uncertainty.py     # MC Dropout uncertainty
│   ├── alert_logic.py     # Alert level determination
│   └── pipeline.py        # Full M4 pipeline
├── m5_architect/          # API + orchestration
│   ├── main.py            # FastAPI app
│   ├── model_runner.py    # Pipeline orchestrator
│   ├── poller.py          # NOAA data poller
│   ├── db.py              # Database layer
│   └── schema.sql         # PostgreSQL schema
├── frontend/
│   └── index.html         # React dashboard (single file)
├── models/                # Model checkpoints
├── data/                  # Training data
├── demo_test.py           # Storm replay demo
├── requirements.txt
├── .env.template
├── start.sh
└── README.md
```

---

## 🧪 Demo: September 2017 Storm

The demo replays the September 6-8, 2017 X9.3 solar flare event:
- **Kp reached 8** (G4 severe storm)
- **Bz dropped to -31 nT** (strongly southward IMF)
- **Solar wind: 770 km/s**
- **GIC estimates: ~18 A/km** (CRITICAL tier)

```bash
python demo_test.py
```

---

## 📊 Output Contracts

All modules communicate via standardized dict structures. See the module docstrings for exact formats.

---

## 📝 License

Academic project — B.E./B.Tech Final Year

## 👤 Author

ICARUS-X Team
