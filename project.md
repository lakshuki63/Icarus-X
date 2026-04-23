╔══════════════════════════════════════════════════════════════════════╗
║           ICARUS-X  ·  MASTER PROJECT PROMPT  ·  ALL 5 MODULES      ║
║         Space Weather Intelligence + GIC Risk Estimation             ║
╚══════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ★  HOW TO USE THIS PROMPT  ★
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: Each member pastes SYSTEM CONTEXT (below) into their AI chat.
STEP 2: Each member also pastes their own MEMBER SECTION (M1–M5).
STEP 3: Ask for ONE FILE AT A TIME. Never ask for the whole module.
STEP 4: Test → fix → next file.
STEP 5: Hand output files to the next member as described.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        SYSTEM CONTEXT
               (ALL MEMBERS PASTE THIS FIRST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are an expert ML engineer helping 2nd year engineering students
build Icarus-X, a real-time space weather forecasting system.

PROJECT: Icarus-X
FULL NAME: Intelligent Coronal Activity Real-time Universal Sentinel eXtended
TEAM: 5 members (Lakshuki, Ness, Nakul, Siddhi, Pranjal)
YEAR: Final year B.E./B.Tech, 2024–2025

WHAT THE SYSTEM DOES:
Takes NASA SDO solar magnetogram images (2D, every 12 min)
+ NOAA DSCOVR solar wind data (1D numbers, every 1 min)
→ Fuses them using deep learning
→ Predicts Kp index (geomagnetic storm scale 0–9) 3–24 hours ahead
→ Translates into GIC (Geomagnetically Induced Current) risk
  for power grid operators: Low / Medium / High / Critical

TECH STACK:
  ML:       Python 3.11, PyTorch 2.x, ultralytics (YOLOv10),
            XGBoost, scikit-learn, imbalanced-learn, scipy
  Backend:  FastAPI, SQLAlchemy, PostgreSQL, Redis, APScheduler
  Frontend: React 18, Recharts, Axios, Tailwind CSS
  Infra:    Docker Compose locally, Render.com free tier for demo
  Training: Kaggle Notebooks (free GPU) or Google Colab

FOLDER STRUCTURE:
  icarus-x/
    m1_visionary/      preprocess.py, annotate.py, train_yolo.py,
                       feature_extractor.py, visionary.py, export_features.py
    m2_predictor/      data_loader.py, windowing.py, model.py,
                       train.py, infer.py, evaluate.py
    m3_classifier/     storm_events.py, features.py, train_xgb.py,
                       infer.py, evaluate.py
    m4_gic/            gic_model.py, uncertainty.py, alert_logic.py,
                       pipeline.py
    m5_architect/      db.py, schema.sql, main.py (FastAPI), poller.py,
                       model_runner.py
    frontend/          src/App.jsx, components/KpChart.jsx,
                       ForecastTable.jsx, GICPanel.jsx,
                       AttentionHeatmap.jsx, AlertBanner.jsx
    requirements.txt
    .env.template
    docker-compose.yml

DATA FLOW (what each module gives the next):
  M1 → ar_features.csv      (timestamp, f0..f11) → M2
  M2 → kp_forecast dict     (8 horizons + CI)    → M4
  M3 → g_tier_probs dict    (G1–G5 probabilities) → M5
  M4 → gic_risk dict        (GIC + risk tier)     → M5
  M5 → REST API + WebSocket                       → React frontend

RULES FOR ALL CODE:
  1. Every file: module docstring at top
  2. Every function: type annotations + one-line docstring
  3. All constants: CONFIG dict at top of each file
  4. All paths: pathlib.Path (never hardcoded strings)
  5. Print progress at every major step
  6. On error: print clearly, skip gracefully (do not crash)
  7. No silent failures anywhere

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

          ██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗███████╗
          ██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝██╔════╝
          ██║ █╗ ██║██║   ██║██████╔╝█████╔╝ ███████╗
          ██║███╗██║██║   ██║██╔══██╗██╔═██╗ ╚════██║
          ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗███████║
           ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          M1 — THE VISIONARY (Solar Image Feature Extraction)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEMBER: [Name of M1 team member]
TRAINING: Kaggle Notebooks (has images already)

WHAT I BUILD:
  Takes NASA SDO/HMI magnetogram images as input.
  Runs YOLOv10 to detect Active Regions (dangerous solar magnetic tangles).
  Runs CNN regression head to extract 12 magnetic complexity features per AR.
  Exports ar_features.csv — the image intelligence for M2.

INPUT I HAVE:
  - Magnetogram images (PNG/JPG) in /kaggle/input/
  - No human bounding box annotations (use automated thresholding)
  - No FITS files needed (already preprocessed to PNG)

OUTPUT I GIVE M2:
  ar_features.csv with columns:
  filename, timestamp, n_regions_detected,
  f0 (total_flux), f1 (pil_length), f2 (strong_field_area),
  f3 (shear_zone), f4 (mean_gradient), f5 (field_asymmetry),
  f6 (compactness), f7 (aspect_ratio), f8 (peak_strength),
  f9 (polarity_mix), f10 (variance), f11 (complexity)

FILES TO BUILD (one at a time, in this order):
  1. preprocess.py
     - Load PNG/JPG images with cv2.imread(GRAYSCALE)
     - Convert uint8 [0,255] → float [-3000, 3000] Gauss proxy
     - Apply signed Gaussian normalisation: (img - mean) / std
     - Clip to [-3, 3]
     - Resize to 512×512 with cv2.INTER_LINEAR
     - Save as PNG with magnetogram_to_png_array() conversion
     - Output: processed_png/ directory

  2. annotate.py
     - Auto-detect ARs using cv2 thresholding + morphology
     - Bright (>200) and dark (<55) uint8 = strong field regions
     - cv2.dilate then cv2.morphologyEx CLOSE to cluster
     - cv2.connectedComponentsWithStats for bounding boxes
     - Filter: area >200px, <90% of image size
     - Add 20px padding to each box
     - Write YOLO format labels: class cx cy w h (normalised 0–1)
     - One .txt per image; empty file = valid negative
     - Output: dataset.yaml + labels/train, val, test splits

  3. train_yolo.py
     - Load YOLOv10n (yolov10n.pt, auto-downloads)
     - Train with:
         imgsz=512, batch=16, epochs=80
         flipud=0.5, fliplr=0.5, degrees=45, scale=0.3
         hsv_h=0.0, hsv_s=0.0, hsv_v=0.0 (grayscale, no colour)
         mosaic=0.0, mixup=0.0 (breaks scale — MUST disable)
         patience=20, workers=2 (Kaggle limit)
     - Save best.pt to /kaggle/working/models/yolov10/
     - Print mAP@0.5 at end

  4. feature_extractor.py
     - Class: ARFeatureHead(nn.Module)
       Input: [batch, 1, H, W] grayscale crop
       Architecture: Conv(1→32) BN ReLU Pool
                   → Conv(32→64) BN ReLU Pool
                   → Conv(64→128) BN ReLU AdaptiveAvgPool(4×4)
                   → Flatten → FC(2048→256) ReLU Dropout(0.3)
                   → FC(256→64) ReLU → FC(64→12)
       Output: [batch, 12] feature vector
     - Function: compute_analytical_features(crop: np.ndarray) → np.ndarray
       Uses cv2 + numpy to compute f0–f11 analytically
       These are pseudo-labels for training the CNN head

  5. train_features.py
     - Dataset: MagnetogramFeatureDataset
       Loads images, takes random 64×64 crops
       Computes analytical features as targets (pseudo-supervision)
     - Train ARFeatureHead 40 epochs
       Loss: MSELoss, Optimizer: AdamW lr=1e-3
       Scheduler: CosineAnnealingLR
     - Save best checkpoint to /kaggle/working/models/feature_head_best.pt
     - Plot + save training curve

  6. visionary.py
     - Class: IcarusVisionary
       __init__(yolo_checkpoint, feature_checkpoint)
       extract(image_path) → dict with bboxes + feature_vector [12]
       extract_batch(image_dir, output_csv) → pd.DataFrame
     - Feature vector = 0.6 × analytical + 0.4 × CNN output
     - If no AR detected: compute analytical on full image
     - One row per image always (zeros if all fail)

  7. export_features.py
     - Instantiate IcarusVisionary
     - Process all images in /kaggle/input/
     - Save ar_features.csv to /kaggle/working/data/
     - Print: total images, mean ARs per image, feature statistics

MY DELIVERABLE TO TEAM:
  ✓ ar_features.csv downloaded from Kaggle working directory
  ✓ Screenshot of YOLOv10 detecting ARs on 5 test images
  ✓ mAP@0.5 score printed (>0.30 is OK for initial stage)
  ✓ feature_head_training.png (loss curve)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          M2 — THE PREDICTOR (Temporal Kp Forecasting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEMBER: [Name of M2 team member]
TRAINING: Kaggle Notebooks

WHAT I BUILD:
  Takes 24-hour rolling window of solar wind data (Bz, Vsw, Np, Pdyn)
  + 12-dim AR feature vector from M1.
  Predicts Kp index at 8 future horizons: 3,6,9,12,15,18,21,24 hours.
  Outputs probabilistic forecast with MC Dropout confidence intervals.

INPUT I RECEIVE FROM M1:
  ar_features.csv (filename, timestamp, f0..f11)

INPUT I SOURCE MYSELF:
  OMNI 1-minute solar wind: https://omniweb.gsfc.nasa.gov
    Columns: YEAR DOY HR MIN Bz_GSM Plasma_Speed Proton_Density Pressure
    Fill values: 9999.99 → replace with NaN
  Kp 3-hourly: https://www.gfz-potsdam.de → upsample to 1-min via ffill
  Download: 2010–2020 for training, 2020 for test

OUTPUT I GIVE M4 AND M5:
  Function: run_forecast() → dict:
  {
    "run_timestamp": "2026-04-22T10:00:00",
    "horizons": [
      {
        "horizon_hr": 3,
        "kp_predicted": 5.2,
        "kp_ci_low": 4.1,
        "kp_ci_high": 6.3,
        "kp_std": 0.7,
        "attention_weights": [...1440 floats...]
      },
      ... (8 total)
    ]
  }

FILES TO BUILD (one at a time):
  1. data_loader.py
     - Load OMNI CSV, parse YEAR+DOY+HR+MIN → datetime index
     - Replace fill values (9999.99, 99999.9, 9999) with NaN
     - Load Kp CSV, upsample to 1-min with ffill
     - Merge on timestamp → columns: [Bz, Vsw, Np, Pdyn, Kp]
     - Save to /data/omni_kp_merged.parquet

  2. normalize.py
     - Load merged parquet
     - Compute mean/std on TRAINING period ONLY: 2010–2017
     - Z-score normalise: Bz, Vsw, Np, Pdyn (NOT Kp — keep 0–9 scale)
     - Save scalers to /models/scalers.pkl with joblib
     - Save normalised parquet to /data/omni_kp_normalised.parquet

  3. windowing.py
     - Function: generate_windows(df, ar_features_df, split)
     - For each valid time T:
         X_sw = normalised solar wind [T-24hr:T] shape [1440, 4]
         X_ar = AR features at nearest timestamp to T (±30min) shape [12]
         y    = Kp at T+3h, T+6h, ..., T+24h  shape [8]
     - Skip window if: gap >3min in input, any target Kp is NaN
     - Splits: train=2010–2017, val=2018–2019, test=2020
     - Save as .npy arrays: X_sw, X_ar, y per split

  4. model.py
     - Class: BahdanauAttention(nn.Module)
       Learns attention weights over encoder hidden states
       Returns context vector [batch, hidden*2] + weights [batch, seq]
     - Class: BiGRUPredictor(nn.Module)
       GRU(bidirectional=True, layers=2, hidden=256, dropout=0.15)
       + BahdanauAttention
       + FusionLayer: concat(context, x_ar) → FC(256) LayerNorm ReLU
       + Decoder: FC(256 → 8)
       Method: mc_dropout_predict(x_sw, x_ar, n_passes=50)
         Keeps model.train() during inference for MC Dropout
         Returns: mean, std, p5, p95 per horizon

  5. train.py
     - AdamW lr=1e-3, weight_decay=1e-4
     - Loss: MSELoss on Kp predictions
     - ReduceLROnPlateau(patience=5, factor=0.5)
     - Gradient clip max_norm=1.0
     - Early stopping: patience=15 on val RMSE at 3hr horizon
     - Log per epoch: train_loss, val_rmse for each of 8 horizons
     - Print RMSE table vs persistence baseline at end of training
     - Save best.pt checkpoint

  6. infer.py
     - Function: run_forecast(solar_wind_df, ar_features, ckpt, scalers)
       Applies scalers to solar_wind_df
       Runs mc_dropout_predict (n_passes=50)
       Returns the standard output dict with 8 horizons + attention
     - This is called by M5 model_runner.py

  7. evaluate.py
     - Load test set (2020), run inference
     - Print RMSE table: horizon | model RMSE | persistence RMSE | Δ%
     - Print: does adding M1 image features improve 3hr RMSE? (ablation)
     - Save: rmse_comparison.csv

MY DELIVERABLE TO TEAM:
  ✓ infer.py working (M5 will import this)
  ✓ best.pt model checkpoint
  ✓ rmse_comparison.csv showing improvement over persistence baseline
  ✓ scalers.pkl (needed by M5 at inference time)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          M3 — THE SENTINEL (Extreme Event Classifier)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEMBER: [Name of M3 team member]
TRAINING: Kaggle Notebooks

WHAT I BUILD:
  Dedicated module for detecting rare severe storms (G3–G5).
  XGBoost with Focal Loss + SMOTE oversampling.
  Outputs per-tier probabilities: G1, G2, G3, G4, G5.

INPUT I SOURCE MYSELF:
  Same OMNI + Kp data as M2 (coordinate to use the same file)
  Kp archive from WDC Kyoto (1975–present) for larger storm library

OUTPUT I GIVE M5:
  Function: classify_window() → dict:
  {
    "g0_prob": 0.12, "g1_prob": 0.35, "g2_prob": 0.28,
    "g3_prob": 0.18, "g4_prob": 0.07,
    "predicted_tier": "G2",
    "top_features": [{"name":"min_bz_6hr","shap_value":0.43}, ...]
  }

FILES TO BUILD (one at a time):
  1. storm_events.py
     - Load Kp time series
     - For every 3-hour window, assign G-tier label:
         G0: max Kp < 5
         G1: 5 ≤ Kp < 6
         G2: 6 ≤ Kp < 7
         G3: 7 ≤ Kp < 8
         G4: Kp ≥ 8
     - Save storm_labels.csv: [window_start, max_kp, g_tier_int 0–4]
     - Print class distribution table with percentages

  2. features.py
     - For each labelled 3-hour window, extract from the 48-hr
       PRECURSOR window (ending at window_start):
         f0: min_bz_6hr       (min Bz in last 6 hours)
         f1: max_vsw_12hr     (max solar wind speed in 12 hours)
         f2: max_dkp_dt_3hr   (max rate of Kp change in 3 hours)
         f3: max_pdyn_6hr     (max dynamic pressure in 6 hours)
         f4: mean_kp_6hr      (mean Kp in 6 hours)
         f5: bz_negative_hrs  (hours with Bz < -5 nT in 12 hours)
     - Save storm_features.csv: [window_start, f0..f5, g_tier_int]

  3. train_xgb.py
     - Load storm_features.csv
     - Train/val/test split: 70/15/15, stratified by g_tier_int
     - Apply SMOTE on training set ONLY
       Oversample G2, G3, G4 to minimum 500 samples each
     - Implement Focal Loss as custom XGBoost objective
       gamma=2.0, alpha=0.25
     - Optuna HPO: 100 trials, optimise F2-score on G3+ (val set)
       Tune: n_estimators, max_depth, learning_rate,
             subsample, colsample_bytree, min_child_weight
     - Train final model with best params
     - Save model to /models/classifier/xgb_best.json

  4. evaluate.py
     - Per G-tier (one-vs-rest):
         TSS = Recall - False Alarm Rate
         HSS = Heidke Skill Score
         F2  = fbeta_score(beta=2) — recall-weighted
         PR-AUC
     - Print results table
     - SHAP summary plot for G3+ class
       Save: shap_g3.png
     - TARGET: TSS > 0.70 for G3+. Print warning if not met.

  5. infer.py
     - Function: classify_window(solar_wind_df, checkpoint)
       Computes f0–f5 from the 48-hour solar wind DataFrame
       Runs XGBoost prediction
       Computes SHAP values for top 3 features
       Returns the standard output dict
     - This is called by M5 model_runner.py

MY DELIVERABLE TO TEAM:
  ✓ infer.py working (M5 will import this)
  ✓ xgb_best.json model checkpoint
  ✓ TSS table (per tier)
  ✓ shap_g3.png (for the report)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          M4 — THE GIC RISK ESTIMATOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEMBER: [Name of M4 team member]
TRAINING: Local or Kaggle (lightweight — no GPU needed)

WHAT I BUILD:
  Converts M2's Kp forecast into real-world GIC risk for power grids.
  Physical formula + MC Dropout neural net + risk tier mapping.
  This is Icarus-X's most novel contribution to the field.

GIC PHYSICS (understand this first):
  Changing magnetic field (dB/dt) → induces electric field (E) in ground
  E acts as voltage source on transmission lines
  GIC = induced current in Amperes per kilometre of line
  GIC ≈ a × exp(b × Kp) + c  ← empirical exponential relationship
  Risk tiers: <1 A/km=Low, 1–5=Medium, 5–15=High, >15=Critical

INPUT I RECEIVE FROM M2:
  run_forecast() output dict with 8 horizons + CI (kp_ci_low/high)

OUTPUT I GIVE M5:
  Function: kp_to_gic_risk() → dict:
  {
    "horizons": [
      {
        "horizon_hr": 3,
        "kp_predicted": 5.2,
        "kp_ci_low": 4.1,
        "kp_ci_high": 6.3,
        "gic_mean": 3.4,
        "gic_p5": 2.1,
        "gic_p95": 5.8,
        "risk_tier": "MEDIUM",
        "risk_tier_worst": "HIGH",
        "risk_color": "#FFA500"
      }, ...8 total
    ],
    "headline_alert": {
      "alert_level": "HIGH",
      "peak_horizon_hr": 9,
      "peak_gic_estimate": 8.3,
      "message": "Geomagnetic storm expected in 9 hours. GIC HIGH.",
      "should_email": True,
      "should_sms": True
    }
  }

FILES TO BUILD (one at a time):
  1. gic_model.py
     - If FINGES data available: load CSV, fit a×exp(b×Kp)+c
       using scipy.optimize.curve_fit(bounds=(0, np.inf))
     - If not available: use fallback params a=0.08, b=0.72, c=0.02
       (from published literature) — print a warning
     - Function: predict_gic(kp, params_path) → float or array
     - Function: get_risk_tier(gic) → str (LOW/MEDIUM/HIGH/CRITICAL)
     - Function: get_risk_color(tier) → str (#hex color)
     - Save params to /models/gic/gic_params.json
     - Plot: Kp vs GIC curve, save as gic_fit.png

  2. uncertainty.py
     - Class: GICNeuralModel(nn.Module)
       Input: Kp scalar normalised to [0,1] (divide by 9)
       Layers: FC(1→32) ReLU Dropout(0.15)
             → FC(32→64) ReLU Dropout(0.15)
             → FC(64→32) ReLU Dropout(0.15)
             → FC(32→1)
     - Train: HuberLoss, AdamW lr=1e-3, 200 epochs
       80/20 train/val split, early stopping patience=20
     - Function: predict_with_uncertainty(kp, checkpoint, n_passes=50)
       model.train() mode at inference (MC Dropout active)
       Run 50 forward passes
       Return: {gic_mean, gic_std, gic_p5, gic_p95,
                risk_tier, risk_tier_worst}

  3. alert_logic.py
     - Function: compute_headline_alert(pipeline_output) → dict
     - Rules (in priority order):
         CRITICAL: any horizon has risk_tier_worst == "CRITICAL"
         HIGH:     any horizon has risk_tier == "HIGH"
         ELEVATED: 3+ consecutive horizons MEDIUM
         WATCH:    any horizon MEDIUM
         NORMAL:   all horizons LOW
     - Returns: alert_level, peak_horizon_hr, peak_gic, message,
                should_email, should_sms

  4. pipeline.py
     - Function: kp_to_gic_risk(kp_forecast, gic_params, gic_nn_ckpt)
       For each of 8 horizons:
         1. Get empirical GIC from gic_model.predict_gic()
         2. Get MC Dropout GIC from uncertainty.predict_with_uncertainty()
         3. Take the wider CI of both methods
         4. Assign risk tiers
       Call compute_headline_alert() to get headline
       Return the full output dict described above
     - This is called by M5 model_runner.py

MY DELIVERABLE TO TEAM:
  ✓ pipeline.py working (M5 will import this)
  ✓ gic_params.json + gic_nn checkpoint
  ✓ gic_fit.png (GIC vs Kp curve for the report)
  ✓ Risk tier logic tested on September 2017 storm replay
    (Kp=8 → should trigger CRITICAL)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          M5 — THE ARCHITECT (API + Database + Dashboard)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEMBER: [Name of M5 team member]
NOTE: NO deployment needed. Run everything locally. Demo is local.

WHAT I BUILD:
  The glue that connects all 4 other modules into one working system.
  FastAPI backend + PostgreSQL database + React frontend.
  Real-time NOAA data polling. WebSocket live updates every 60 seconds.

WHAT I RECEIVE FROM EACH MEMBER:
  From M1: ar_features.csv
  From M2: infer.py (run_forecast function) + best.pt + scalers.pkl
  From M3: infer.py (classify_window function) + xgb_best.json
  From M4: pipeline.py (kp_to_gic_risk function) + gic_params.json

FILES TO BUILD (one at a time):
  1. schema.sql
     CREATE TABLE solar_wind (
       timestamp TIMESTAMPTZ PRIMARY KEY,
       bz FLOAT, vsw FLOAT, np FLOAT, pdyn FLOAT, bt FLOAT,
       source VARCHAR(20) DEFAULT 'DSCOVR'
     );
     CREATE TABLE kp_index (
       timestamp TIMESTAMPTZ PRIMARY KEY,
       kp_value FLOAT NOT NULL, source VARCHAR(20) DEFAULT 'GFZ'
     );
     CREATE TABLE model_forecasts (
       id SERIAL PRIMARY KEY, run_timestamp TIMESTAMPTZ NOT NULL,
       horizon_hr INT NOT NULL,
       kp_predicted FLOAT, kp_ci_low FLOAT, kp_ci_high FLOAT,
       gic_mean FLOAT, gic_p95 FLOAT,
       risk_tier VARCHAR(10), risk_tier_worst VARCHAR(10),
       g_tier_probs JSONB, attention_weights JSONB
     );
     CREATE TABLE storm_alerts (
       id SERIAL PRIMARY KEY, triggered_at TIMESTAMPTZ DEFAULT NOW(),
       alert_level VARCHAR(10), peak_horizon_hr INT,
       peak_gic FLOAT, message TEXT, acknowledged BOOLEAN DEFAULT FALSE
     );

  2. db.py
     - SQLAlchemy ORM models for all 4 tables
     - engine = create_engine(os.environ["DATABASE_URL"])
     - SessionLocal factory
     - get_db() FastAPI dependency
     - init_db() runs schema.sql on first startup

  3. ingest.py
     - Load historical OMNI + Kp CSV files
     - Bulk insert into solar_wind and kp_index tables
     - Print row counts after insert

  4. poller.py
     - Function: poll_noaa_dscovr(db_session)
       Fetch from two NOAA endpoints:
         MAG:    https://services.swpc.noaa.gov/products/solar-wind/mag-1-m.json
         PLASMA: https://services.swpc.noaa.gov/products/solar-wind/plasma-1-m.json
       Parse: timestamp, Bz, Vsw, Np, compute Pdyn = 1.67e-6 × Np × Vsw²
       INSERT OR IGNORE on timestamp conflict
       Timeout: 10 seconds. On error: log and skip.
     - Schedule with APScheduler every 1 minute

  5. model_runner.py
     - Function: run_full_pipeline(db_session) → dict
       1. Query last 1440 rows from solar_wind table → DataFrame
       2. Load latest row from ar_features.csv → 12-dim array
       3. Call m2_predictor.infer.run_forecast()
       4. Call m3_classifier.infer.classify_window()
       5. Call m4_gic.pipeline.kp_to_gic_risk()
       6. Merge g_tier_probs into output dict
       7. Save all 8 horizons to model_forecasts table
       8. If alert_level >= HIGH: insert into storm_alerts
       9. Return complete output dict
     - Schedule with APScheduler every 3 hours
     - Wrap in try/except: log error, return None on failure

  6. main.py (FastAPI — all endpoints)
     GET  /api/health
       → {status, uptime_seconds, last_solar_wind_timestamp}
     GET  /api/solarwind/latest?minutes=60
       → list of solar_wind rows
     GET  /api/forecast/full
       → calls run_full_pipeline(), returns complete dict
     GET  /api/forecast/{horizon_hr}
       → single horizon from latest run
     GET  /api/storms/history?days=30
       → all storm_alerts from last N days
     GET  /api/attention/{run_id}
       → attention_weights JSON for one run
     POST /api/alerts/config
       → save user alert thresholds to DB
     WS   /ws/live
       → send latest forecast immediately on connect
       → send update every 60 seconds
     App config: CORS allow all origins, startup: init_db() + schedulers

  7. React Frontend components (one at a time):
     App.jsx:
       - WebSocket connection to ws://localhost:8000/ws/live
       - Store forecast in useState, update on WS message
       - Fetch /api/solarwind/latest on mount
       - Layout: AlertBanner top, then KpChart, ForecastTable, GICPanel, AttentionHeatmap

     KpChart.jsx:
       - Recharts ComposedChart
       - Line: historical Kp (last 24hr, gray)
       - Line: predicted Kp for 8 horizons (colored by tier)
       - Area: CI band (kp_ci_low to kp_ci_high, semi-transparent)
       - X-axis: -24hr to +24hr with NOW marker
       - Y-axis: 0–9 with G1/G2/G3/G4 threshold lines
       - Legend showing G-tier thresholds

     ForecastTable.jsx:
       - 8-row table: Lead Time | Kp | CI | G-Tier Prob | GIC | Risk
       - Risk column: colored badge (green/yellow/orange/red)
       - Row tinted by risk tier

     GICPanel.jsx:
       - Large display: alert_level in big bold text
       - Peak GIC estimate in A/km
       - Peak horizon in hours
       - One plain-English message

     AttentionHeatmap.jsx:
       - 1440 attention weights → downsample to 24 hourly buckets
       - 24 colored cells: white (low) to deep blue (high)
       - Label: "Hours before forecast: -24h ... now"
       - Tooltip showing exact weight per hour

     AlertBanner.jsx:
       - Fixed at top of page when alert_level HIGH or CRITICAL
       - RED background for CRITICAL, ORANGE for HIGH
       - Dismissible with X button

MY DELIVERABLE TO TEAM:
  ✓ FastAPI running at localhost:8000 with all endpoints working
  ✓ React frontend at localhost:3000 showing live data
  ✓ NOAA poller running (solar wind updating in DB every minute)
  ✓ Full pipeline test: Sep 2017 storm replay shows CRITICAL alert
  ✓ API latency < 500ms (measure with curl timing)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    INTEGRATION TIMELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Week 1 Days 1–3:  All members build their own files independently
Week 1 Day 4:     M1 hands ar_features.csv to M2
                  M2+M1 integrate: retrain Bi-GRU with image features
                  M4 tests pipeline with mock Kp input
Week 1 Day 5:     M5 wires M2+M3+M4 into model_runner.py
                  Full pipeline test on Sep 2017 storm replay
Week 2 Days 1–3:  Evaluation metrics per module
Week 2 Days 4–5:  React frontend complete + load testing
Week 2 Final:     Demo video (10 min showing Sep 2017 storm replay)
                  Final report written from evaluation numbers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                 IF SOMETHING BREAKS — DEBUG ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Print X.shape and y.shape before every model call — wrong shapes
   are the #1 cause of errors in PyTorch.
2. Print df.isna().sum() or tensor.isnan().any() to check for NaN.
3. Is the model loaded? Print type(model) + model.training after load.
4. Is the DB connected? Run:
   python -c "from m5_architect.db import engine; print(engine.url)"
5. Is the API responding?
   curl http://localhost:8000/api/health
6. Is React reaching API?
   Open browser DevTools → Network tab → check the fetch calls.
7. Paste the EXACT error + full stack trace into AI.
   Never describe errors — always paste them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                   ★  END OF MASTER PROMPT  ★
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━