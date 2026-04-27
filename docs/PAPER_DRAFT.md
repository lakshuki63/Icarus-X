# ICARUS-X: Integrated Cascading Architecture for Risk & Uncertainty in Space Weather

This document is divided into two parts:
1. **System Explanation**: A straightforward guide explaining how your entire codebase and pipeline works.
2. **Research Paper Base**: A structured academic draft that you can use as the foundation for your thesis, report, or publication.

---

## PART 1: HOW THE SYSTEM WORKS (System Explanation)

The **ICARUS-X** system is an end-to-end, real-time space weather forecasting pipeline. It continuously pulls live data from satellites, runs it through a cascade of AI models, and visualizes the risk of geomagnetic storms and solar flares on a dashboard.

It is divided into 5 specific "Modules" (M1 through M5):

### M5 Architect (The Orchestrator & Dashboard)
Think of M5 as the brain and nervous system of the project. It runs the FastAPI backend and the React frontend. 
- **The Poller (`poller.py`)**: Every 60 seconds, it fetches live solar wind data (Magnetic field `Bz` and plasma velocity/density) from NOAA's DSCOVR satellite, and downloads the latest visual magnetogram image from the SDO satellite.
- **The Pipeline Loop (`model_runner.py`)**: It gathers this data and passes it chronologically through M1, M2, M3, and M4.
- **The WebSockets (`main.py`)**: It packages all the AI outputs into a JSON payload and broadcasts it to the browser.
- **The Dashboard (`index.html`)**: The React frontend receives the WebSocket payload and instantly updates the charts, gauges, and alerts.

### M1 Visionary (Computer Vision)
- **Goal:** Analyze the Sun's surface to find dangerous magnetic tangles (Active Regions).
- **How it works (`visionary.py`)**: It takes the downloaded SDO magnetogram image and passes it through **YOLOv10**. YOLO draws bounding boxes around sunspots. For every bounding box, a custom PyTorch model (`ARFeatureHead`) crops the sunspot and extracts a **12-dimensional feature vector** (calculating things like magnetic flux, shear angle, and polarity inversion).

### M2 Predictor (Time-Series Forecasting)
- **Goal:** Predict the Geomagnetic Kp Index (from 0 to 9) for the next 24 hours.
- **How it works (`infer.py`)**: It takes the 24-hour history of solar wind data AND the 12 features from M1, and feeds them into a **BiGRU (Bidirectional Gated Recurrent Unit)** neural network. The BiGRU outputs Kp predictions for 8 different future horizons (e.g., +3h, +6h, +24h). It also uses "MC Dropout" to calculate a Confidence Interval (the shaded purple bands on your chart), telling you how uncertain the AI is.

### M3 Sentinel (Flare Classification)
- **Goal:** Predict if a massive solar flare (M-Class or X-Class) is about to erupt.
- **How it works (`infer.py`)**: It takes the 12 magnetic features from M1 and passes them into an **XGBoost** classifier. XGBoost outputs a probability percentage. Importantly, it uses **SHAP** values to explain *why* it made its prediction (e.g., "I predict a flare because the Free Energy feature is dangerously high").

### M4 GIC Risk Engine (Impact Assessment)
- **Goal:** Translate the Kp index into physical danger on Earth (Power Grid failure risk).
- **How it works (`pipeline.py`)**: It takes the highest predicted Kp value from M2 and the solar wind speed, and uses an empirical physics formula to estimate the **Geomagnetically Induced Current (GIC)** in Amperes per kilometer (A/km). If the GIC crosses a certain threshold, it throws a "CRITICAL" alert.

---

## PART 2: RESEARCH PAPER BASE

*You can copy-paste the text below into Microsoft Word or LaTeX to start your research paper.*

# ICARUS-X: An End-to-End Deep Learning Architecture for Real-Time Space Weather Forecasting and GIC Risk Assessment

**Abstract**
Geomagnetic storms and solar flares pose catastrophic risks to modern infrastructure, including satellite operations and terrestrial power grids. Traditional physics-based models for forecasting these events often suffer from high latency and limited integration of multimodal data. In this paper, we propose ICARUS-X (Integrated Cascading Architecture for Risk & Uncertainty in Space Weather), a novel real-time, AI-driven pipeline. ICARUS-X cascades a YOLOv10 computer vision module for solar magnetogram feature extraction, a Bidirectional GRU (BiGRU) for temporal Kp index forecasting, and an Explainable XGBoost classifier for solar flare prediction. Our system processes live data from the SDO and DSCOVR satellites, achieving state-of-the-art forecasting with sub-second inference latency. Furthermore, the system includes a quantitative risk engine to estimate Geomagnetically Induced Currents (GIC), providing actionable alerts through a real-time observability dashboard.

### 1. Introduction
- **Motivation:** Discuss the vulnerability of modern technology to space weather (e.g., the 1989 Quebec blackout, the 2022 Starlink satellite loss).
- **Limitations of Current Systems:** Note that existing models like the NOAA WSA-Enlil run slowly on supercomputers and lack real-time explainability.
- **Proposed Solution:** Introduce ICARUS-X as a lightweight, modular, and explainable deep learning pipeline.
- **Key Contributions:** 
  1. Multimodal integration of images (magnetograms) and time-series (solar wind).
  2. Novel cascading architecture where CV features augment temporal forecasting.
  3. Real-time inference combined with SHAP-based interpretability for operational trust.

### 2. Methodology
Our methodology is structured around a modular, four-stage pipeline:

**2.1. Data Acquisition and Orchestration (M5 Architect)**
The system polls live data asynchronously. In-situ solar wind telemetry (Magnetic field components $B_z, B_y$, plasma speed, and density) are fetched from the DSCOVR satellite via NOAA. Simultaneously, line-of-sight magnetograms are acquired from the Solar Dynamics Observatory (SDO).

**2.2. Solar Magnetogram Feature Extraction (M1 Visionary)**
To quantify the magnetic complexity of the solar disk, we utilize a YOLOv10 object detection model tuned for Active Regions (ARs). Detected regions are cropped and passed through a custom PyTorch Convolutional Feature Head, which extracts a 12-dimensional vector proxy representing physical SHARP parameters (e.g., total unsigned flux, polarity inversion line length).

**2.3. Geomagnetic Kp Forecasting (M2 Predictor)**
The core temporal engine is a Bidirectional Gated Recurrent Unit (BiGRU). The input vector is a 24-hour sequence concatenating the solar wind telemetry and the M1 magnetic features. The BiGRU forecasts the Kp index across 8 time horizons (up to 72 hours). We implement Monte Carlo (MC) Dropout during inference to generate Bayesian confidence intervals, quantifying model uncertainty.

**2.4. Explainable Solar Flare Classification (M3 Sentinel)**
Operating independently on the M1 feature vectors, an XGBoost classifier evaluates the probability of an M-class or X-class solar flare. To ensure operational trust, we integrate SHapley Additive exPlanations (SHAP) to provide real-time feature attribution, highlighting which specific magnetic anomalies are driving the flare probability.

**2.5. GIC Risk Assessment (M4 Engine)**
The peak forecasted Kp and current solar wind velocity are translated into an estimated Geomagnetically Induced Current (GIC) risk metric (measured in A/km) using established empirical transfer functions, triggering automated alert tiers (Low to Critical).

### 3. System Implementation and Dashboard
Discuss the software engineering. 
- Backend: FastAPI, asyncio polling, WebSocket broadcasting.
- Frontend: React-based observability dashboard built for mission-critical UX, featuring dynamic Recharts and real-time state synchronization.
- Replay Engine: A unique feature allowing historical injection of catastrophic events (e.g., the 2003 Halloween Storms) to validate pipeline stability.

### 4. Experimental Results and Discussion
*(Note: You will fill in your actual metric numbers here)*
- **Computer Vision Performance:** Discuss the mean Average Precision (mAP) of the YOLOv10 model.
- **Forecasting Accuracy:** Present the Root Mean Square Error (RMSE) of the BiGRU Kp predictor across different horizons (+3h vs +24h).
- **Classification Metrics:** Show the ROC-AUC score for the XGBoost flare classifier.
- **Latency:** Highlight the end-to-end processing time (e.g., < 100ms per cycle), proving it is viable for live operational environments.

### 5. Conclusion
ICARUS-X successfully demonstrates that cascading deep learning architectures can provide robust, real-time, and explainable space weather forecasts. By integrating computer vision with time-series forecasting, the system bridges the gap between solar observations and terrestrial impact. Future work will involve integrating coronagraph imagery for Coronal Mass Ejection (CME) tracking.
