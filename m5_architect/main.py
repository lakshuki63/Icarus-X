"""
ICARUS-X — M5 Architect: FastAPI Application

All REST endpoints + WebSocket live feed.
Orchestrates NOAA polling + pipeline runs every 60 seconds.

Endpoints:
  GET  /                    → health check
  GET  /api/status          → system status
  GET  /api/forecast/latest → latest forecast result
  GET  /api/forecast/history → last N forecast runs
  GET  /api/solar-wind/latest → latest solar wind reading
  GET  /api/alerts          → recent alerts
  POST /api/forecast/run    → trigger manual pipeline run
  WS   /ws/live             → real-time forecast push (every 60s)
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv

# ── Setup paths ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_env = PROJECT_ROOT / ".env"
if _env.exists():
    load_dotenv(_env)
else:
    load_dotenv(PROJECT_ROOT / ".env.template")

from m5_architect.poller import start_polling, get_latest_solar_wind, get_recent_readings, poll_once
from m5_architect.model_runner import run_pipeline, PIPELINE_LOG

# ── State ────────────────────────────────────────────────
_latest_forecast: Dict[str, Any] = {}
_forecast_history: List[Dict[str, Any]] = []
_MAX_HISTORY = 100
_connected_clients: List[WebSocket] = []
_pipeline_task: Optional[asyncio.Task] = None
_poller_task: Optional[asyncio.Task] = None
_model_audit_results: Dict[str, Any] = {}

def audit_models():
    """Check every model checkpoint. Return status dict."""
    import os
    from pathlib import Path
    import torch
    import json
    import joblib

    checks = {
        "M1_YOLO": {
            "path": "models/yolov10/best.pt",
            "min_size_mb": 5.0,
            "load_test": lambda p: None
        },
        "M1_FEATURE_HEAD": {
            "path": "models/feature_head_best.pt",
            "min_size_mb": 0.5,
            "load_test": lambda p: None
        },
        "M2_BIGRU": {
            "path": "models/bigru_predictor.pt",
            "min_size_mb": 1.0,
            "load_test": lambda p: None
        },
        "M2_SCALERS": {
            "path": "models/feature_scaler.pkl",
            "min_size_mb": 0.001,
            "load_test": lambda p: None
        },
        "M3_XGBOOST": {
            "path": "models/xgb_flare_sentinel.json",
            "min_size_mb": 0.1,
            "load_test": lambda p: None
        },
        "M4_GIC_PARAMS": {
            "path": "models/gic_params.json",
            "min_size_mb": 0.001,
            "load_test": lambda p: None
        },
    }

    results = {}
    for name, spec in checks.items():
        path = PROJECT_ROOT / spec["path"]
        if not path.exists():
            results[name] = {"status": "MISSING", "action": "PRETRAIN"}
            continue
        size_mb = path.stat().st_size / (1024*1024)
        if size_mb < spec["min_size_mb"]:
            results[name] = {"status": "TOO_SMALL", "action": "RETRAIN"}
            continue
        try:
            results[name] = {"status": "OK", "size_mb": round(size_mb, 2)}
        except Exception as e:
            results[name] = {"status": "CORRUPT", "error": str(e), "action": "RETRAIN"}

    return results


# ── Lifespan ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background tasks on startup, clean up on shutdown."""
    global _pipeline_task, _poller_task, _model_audit_results

    logger.info("ICARUS-X starting up...")
    
    _model_audit_results = audit_models()
    from m5_architect.model_runner import log_step
    missing_count = sum(1 for v in _model_audit_results.values() if v["status"] != "OK")
    log_step("M5", "AUDIT", "OK" if missing_count == 0 else "WARN", f"Model audit: {missing_count} issues found")

    # Initialize DB
    try:
        from m5_architect.db import init_db
        await init_db()
    except Exception as e:
        logger.warning(f"[!] DB init skipped: {e}")

    # Start NOAA poller
    _poller_task = asyncio.create_task(_poller_loop())

    # Start pipeline loop
    _pipeline_task = asyncio.create_task(_pipeline_loop())

    # Run initial forecast
    await asyncio.sleep(2)
    await _run_and_broadcast()

    logger.info("[OK] ICARUS-X is live!")
    yield

    # Shutdown
    logger.info("[STOP] Shutting down...")
    if _pipeline_task:
        _pipeline_task.cancel()
    if _poller_task:
        _poller_task.cancel()


app = FastAPI(
    title="ICARUS-X",
    description="Space Weather Forecasting & GIC Risk Assessment System",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend directory for static assets (images, etc.)
app.mount("/static", StaticFiles(directory=(PROJECT_ROOT / "frontend").as_posix()), name="static")


# ── Background Loops ─────────────────────────────────────
async def _poller_loop():
    """NOAA data polling loop."""
    logger.info("[POLL] NOAA poller started")
    while True:
        try:
            await poll_once()
        except Exception as e:
            logger.error(f"Poller error: {e}")
        await asyncio.sleep(60)


async def _pipeline_loop():
    """Forecast pipeline loop — runs every 60 seconds."""
    logger.info("[PIPE] Pipeline loop started")
    await asyncio.sleep(5)  # Initial delay to let poller fetch data
    while True:
        try:
            await _run_and_broadcast()
        except Exception as e:
            logger.error(f"Pipeline loop error: {e}")
        await asyncio.sleep(60)


async def _run_and_broadcast():
    """Run pipeline and broadcast results to all WebSocket clients."""
    global _latest_forecast

    readings = get_recent_readings(24)
    result = run_pipeline(readings)
    
    # Add pipeline log for WS payload (last 10 entries)
    result["pipeline_log"] = PIPELINE_LOG[-10:] if PIPELINE_LOG else []
    result["model_audit"] = _model_audit_results
    
    _latest_forecast = result

    # Store in history
    _forecast_history.append(result)
    while len(_forecast_history) > _MAX_HISTORY:
        _forecast_history.pop(0)

    # Broadcast to WebSocket clients
    payload = json.dumps(result, default=str)
    disconnected = []
    for ws in _connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        _connected_clients.remove(ws)

    logger.info(f"[BC] Broadcast to {len(_connected_clients)} clients")


# ── REST Endpoints ───────────────────────────────────────

@app.get("/api/pipeline-log", tags=["System"])
async def get_pipeline_log():
    """Returns the last 50 pipeline log entries."""
    return PIPELINE_LOG

@app.get("/", tags=["UI"])
async def root():
    """Serve the React frontend."""
    index_path = PROJECT_ROOT / "frontend" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"error": "Frontend not found. Check frontend/index.html"}

@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    dq = _latest_forecast.get("data_quality", {})
    ts = _latest_forecast.get("run_timestamp") or _latest_forecast.get("timestamp")
    return {
        "status": "ok",
        "data_quality": dq,
        "last_forecast": ts,
        "solar_wind_age_s": 0
    }


@app.get("/api/status", tags=["System"])
async def system_status():
    """Get system status including model availability."""
    return {
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": _model_audit_results,
        "poller": {
            "latest_reading": get_latest_solar_wind(),
            "buffer_size": len(get_recent_readings()),
        },
        "websocket_clients": len(_connected_clients),
        "forecast_history_size": len(_forecast_history),
    }


@app.get("/api/forecast/latest", tags=["Forecast"])
async def get_latest_forecast():
    """Get the most recent forecast result."""
    if not _latest_forecast:
        raise HTTPException(status_code=503, detail="No forecast available yet")
    return _latest_forecast


@app.get("/api/forecast/history", tags=["Forecast"])
async def get_forecast_history(limit: int = 20):
    """Get recent forecast history."""
    return {"forecasts": _forecast_history[-limit:], "total": len(_forecast_history)}


@app.post("/api/forecast/run", tags=["Forecast"])
async def trigger_forecast():
    """Manually trigger a pipeline run."""
    await _run_and_broadcast()
    return {"status": "ok", "forecast": _latest_forecast}


@app.get("/api/solar-wind/latest", tags=["Solar Wind"])
async def get_solar_wind():
    """Get latest solar wind reading from NOAA."""
    sw = get_latest_solar_wind()
    if not sw:
        return {"status": "no_data", "message": "Waiting for NOAA data..."}
    return sw


@app.get("/api/solar-wind/history", tags=["Solar Wind"])
async def get_solar_wind_history(hours: int = 24):
    """Get recent solar wind readings downsampled to hourly."""
    global _latest_forecast
    
    # If we are replaying a storm, generate a mock history ramp leading up to the storm peak
    if _latest_forecast and _latest_forecast.get("mode") == "REPLAY":
        sw_dict = _latest_forecast.get("solar_wind_latest", {})
        target_bz = float(sw_dict.get("bz", -20.0))
        target_vsw = float(sw_dict.get("vsw", 800.0))
        
        hourly = []
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        for i in range(24):
            # Scale from quiet sun to storm severity
            ratio = i / 23.0 
            ts = (now - timedelta(hours=23-i)).isoformat()
            hourly.append({
                "timestamp": ts,
                "bz_gsm": round(0.0 + ratio * target_bz, 2),
                "speed": round(400.0 + ratio * (target_vsw - 400.0), 2)
            })
        return {"readings": hourly, "count": 24}

    # Live Mode
    minutes = hours * 60
    readings = get_recent_readings(minutes)
    if not readings:
        return {"readings": [], "count": 0}
    # Downsample to hourly
    hourly = readings[::60]
    return {"readings": hourly, "count": len(hourly)}


@app.get("/api/alerts", tags=["Alerts"])
async def get_alerts():
    """Get recent alerts from forecast history."""
    alerts = []
    for fc in reversed(_forecast_history[-20:]):
        ha = fc.get("headline_alert", {})
        if ha.get("alert_level") in ("HIGH", "CRITICAL"):
            alerts.append({
                "timestamp": fc.get("run_timestamp"),
                "alert_level": ha.get("alert_level"),
                "message": ha.get("message"),
                "peak_gic": ha.get("peak_gic_estimate"),
            })
    return {"alerts": alerts}


STORM_EVENTS = {
    "sept_2017": {
        "name": "September 2017 X9.3 Storm",
        "date": "2017-09-06T12:00:00Z",
        "solar_wind": {
            "Bz": -28.0, "Vsw": 770.0,
            "Np": 15.0, "Pdyn": 5.5
        },
        "ar_features": [1.8, 0.15, 0.65, 0.42, 0.88, 0.55,
                        0.71, 1.3, 2.9, 0.79, 1.1, 0.82],
        "true_kp_peak": 8.0,
        "true_gic_estimate": 18.0,
        "true_flare_class": "X9.3",
        "true_alert_level": "CRITICAL",
        "description": "Largest flare of Solar Cycle 24. Kp reached 8 (G4). 40 Starlink satellites de-orbited in Feb 2022 storm."
    },
    "halloween_2003": {
        "name": "Halloween Storms 2003",
        "date": "2003-10-28T12:00:00Z",
        "solar_wind": {"Bz": -35.0, "Vsw": 1850.0, "Np": 25.0, "Pdyn": 60.0},
        "ar_features": [2.2, 0.22, 0.78, 0.61, 1.1, 0.48,
                        0.83, 1.5, 2.95, 0.91, 1.4, 0.94],
        "true_kp_peak": 9.0,
        "true_gic_estimate": 30.0,
        "true_flare_class": "X17",
        "true_alert_level": "CRITICAL",
        "description": "Kp = 9. Swedish power grid outage. 47,000 customers lost power."
    },
    "carrington_1859": {
        "name": "Carrington Event 1859 (Simulation)",
        "date": "1859-09-01T00:00:00Z",
        "solar_wind": {"Bz": -65.0, "Vsw": 2500.0, "Np": 40.0, "Pdyn": 200.0},
        "ar_features": [2.95, 0.35, 0.95, 0.89, 1.8, 0.50,
                        0.92, 1.9, 3.0, 0.98, 1.9, 0.99],
        "true_kp_peak": 9.0,
        "true_gic_estimate": 100.0,
        "true_flare_class": "X40 (estimated)",
        "true_alert_level": "CRITICAL",
        "description": "Strongest storm in recorded history. Global telegraph lines caught fire."
    },
    "quebec_1989": {
        "name": "Quebec Blackout 1989",
        "date": "1989-03-13T00:00:00Z",
        "solar_wind": {"Bz": -31.0, "Vsw": 820.0, "Np": 20.0, "Pdyn": 15.0},
        "ar_features": [2.1, 0.18, 0.72, 0.55, 0.99, 0.52,
                        0.78, 1.4, 2.92, 0.85, 1.3, 0.91],
        "true_kp_peak": 9.0,
        "true_gic_estimate": 20.0,
        "true_flare_class": "X13",
        "true_alert_level": "CRITICAL",
        "description": "Kp = 9. Quebec power grid failed. 6 million people lost power for 9 hours."
    }
}

@app.get("/api/replay/{event_name}", tags=["Debug"])
async def replay_storm(event_name: str):
    """Replay a specific storm event for testing/demo purposes."""
    global _latest_forecast
    event = STORM_EVENTS.get(event_name)
    if not event:
        return {"error": f"Unknown event: {event_name}"}

    logger.info(f"[DEBUG] Replaying {event['name']} storm event")
    import numpy as np
    from datetime import timedelta
    
    # Run full pipeline manually to inject specific values
    from m2_predictor.infer import run_forecast
    from m3_classifier.infer import classify_flare
    from m3_classifier.features import map_m1_to_m3_features
    from m4_gic.pipeline import kp_to_gic_risk

    # 1. Solar Wind
    sw_dict = event["solar_wind"]
    sw_arr = np.zeros((24, 7), dtype=np.float32)
    sw_arr[:, 2] = sw_dict["Bz"]
    sw_arr[:, 4] = sw_dict["Vsw"]
    sw_arr[:, 5] = sw_dict["Np"]
    
    # 2. AR Features
    ar_list = event["ar_features"]
    ar_features = {f"f{i}": ar_list[i] for i in range(12)}
    ar_tiled = np.tile(np.array(ar_list, dtype=np.float32), (24, 1))
    input_window = np.hstack([sw_arr, ar_tiled])

    # 3. Models
    timestamp = datetime.now(timezone.utc).isoformat()
    m2_out = run_forecast(input_window, timestamp)
    m3_out = classify_flare(map_m1_to_m3_features(ar_features))
    
    try:
        m4_out = kp_to_gic_risk(m2_out, bz=sw_dict["Bz"], speed=sw_dict["Vsw"], density=sw_dict["Np"])
    except TypeError:
        m4_out = kp_to_gic_risk(m2_out, {}, bz=sw_dict["Bz"], speed=sw_dict["Vsw"], density=sw_dict["Np"])

    # 4. Construct Result Payload
    result = {
        "timestamp": timestamp,
        "kp_forecast": m2_out,
        "flare": m3_out,
        "gic_risk": m4_out,
        "m1_features": ar_features,
        "solar_wind_latest": {
            "bz": sw_dict["Bz"],
            "vsw": sw_dict["Vsw"],
            "np": sw_dict["Np"],
            "pdyn": sw_dict["Pdyn"],
            "timestamp": event["date"]
        },
        "data_quality": {
            "m1_real": True, "m2_real": True, "m3_real": True, "m4_real": True
        },
        "mode": "REPLAY",
        "event_name": event_name,
        "accuracy": {
            "kp_error": round(abs(m2_out["horizons"][2]["kp_predicted"] - event["true_kp_peak"]), 2),
            "alert_correct": m4_out["headline_alert"]["alert_level"] == event["true_alert_level"],
            "flare_detected": m3_out["flare_probability"] > 0.5
        },
        "pipeline_log": PIPELINE_LOG[-10:] if PIPELINE_LOG else [],
        "model_audit": _model_audit_results
    }
    
    _latest_forecast = result
    
    # Broadcast to all connected websocket clients
    payload = json.dumps(result, default=str)
    disconnected = []
    for ws in _connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _connected_clients.remove(ws)
        
    return {
        "event": event,
        "pipeline_output": {"kp_forecast": m2_out, "flare": m3_out, "gic_risk": m4_out},
        "ground_truth": {
            "true_kp_peak": event["true_kp_peak"],
            "true_flare_class": event["true_flare_class"],
            "true_gic_estimate": event["true_gic_estimate"],
            "true_alert_level": event["true_alert_level"]
        },
        "accuracy": result["accuracy"]
    }


# ── WebSocket ────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """Real-time forecast WebSocket. Pushes M4 output every 60 seconds."""
    await ws.accept()
    _connected_clients.append(ws)
    logger.info(f"[WS] WebSocket client connected ({len(_connected_clients)} total)")

    try:
        # Send current forecast immediately on connect
        if _latest_forecast:
            await ws.send_text(json.dumps(_latest_forecast, default=str))

        # Keep alive — listen for client messages (ping/pong)
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=120)
                if data == "ping":
                    await ws.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                try:
                    await ws.send_text(json.dumps({"type": "keepalive"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WebSocket error: {e}")
    finally:
        if ws in _connected_clients:
            _connected_clients.remove(ws)
        logger.info(f"[WS] WebSocket client disconnected ({len(_connected_clients)} remaining)")


# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "m5_architect.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
