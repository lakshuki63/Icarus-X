"""
ICARUS-X — M5 Architect: NOAA Data Poller
Hits NOAA DSCOVR (JSON) and Kp (Text/JSON) endpoints every 60s.
Maintains an in-memory buffer of the last 24-48 hours.
"""

import asyncio
import httpx
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger

# ── NOAA Endpoints ───────────────────────────────────────
MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
KP_URL = "https://services.swpc.noaa.gov/products/noaa-estimated-planetary-k-index-1-minute.json"

# ── State ────────────────────────────────────────────────
_solar_wind_buffer: List[Dict[str, Any]] = []
_kp_buffer: List[Dict[str, Any]] = []
_MAX_BUFFER_HOURS = 48


async def poll_once():
    """Fetch latest data from NOAA and update buffers."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # 1. Fetch Mag
            mag_resp = await client.get(MAG_URL)
            mag_data = mag_resp.json() if mag_resp.status_code == 200 else []
            
            # 2. Fetch Plasma
            plasma_resp = await client.get(PLASMA_URL)
            plasma_data = plasma_resp.json() if plasma_resp.status_code == 200 else []
            
            # 3. Fetch Kp
            kp_resp = await client.get(KP_URL)
            kp_data = kp_resp.json() if kp_resp.status_code == 200 else []

            # Process and merge (Simplified for demo — in prod we'd align timestamps precisely)
            if mag_data and plasma_data:
                _process_noaa_data(mag_data, plasma_data)
            
            if kp_data:
                _process_kp_data(kp_data)

            logger.debug(f"[POLL] NOAA updated. SW Buffer: {len(_solar_wind_buffer)}, Kp Buffer: {len(_kp_buffer)}")
        except Exception as e:
            logger.error(f"[POLL] Fetch failed: {e}")


def _process_noaa_data(mag: List[List[Any]], plasma: List[List[Any]]):
    """Convert NOAA JSON lists to flat dicts and merge."""
    global _solar_wind_buffer
    
    # NOAA format: [ ["time_tag", "bx", ...], ["2024-...", "1.2", ...] ]
    if len(mag) < 2 or len(plasma) < 2:
        return

    m_headers = mag[0]
    p_headers = plasma[0]
    
    # Just take the last 100 points for simplicity in this stub
    new_readings = []
    for i in range(1, min(len(mag), len(plasma))):
        m_row = mag[i]
        p_row = plasma[i]
        
        # Simple alignment check
        if m_row[0] != p_row[0]: continue
        
        reading = {
            "timestamp": m_row[0],
            "bx_gsm": float(m_row[1]) if m_row[1] else 0.0,
            "by_gsm": float(m_row[2]) if m_row[2] else 0.0,
            "bz_gsm": float(m_row[3]) if m_row[3] else 0.0,
            "bt": float(m_row[6]) if m_row[6] else 0.0,
            "density": float(p_row[1]) if p_row[1] else 0.0,
            "speed": float(p_row[2]) if p_row[2] else 0.0,
            "temperature": float(p_row[3]) if p_row[3] else 0.0,
        }
        new_readings.append(reading)
    
    _solar_wind_buffer = new_readings[-1440:] # Keep 24 hours @ 1-min


def _process_kp_data(kp: List[Dict[str, Any]]):
    """Process Kp index list."""
    global _kp_buffer
    # NOAA format for 1-min Kp is a list of dicts
    # [{"time_tag": "...", "estimated_kp": "..."}]
    processed = []
    for row in kp:
        processed.append({
            "timestamp": row["time_tag"],
            "kp_value": float(row["estimated_kp"]),
        })
    _kp_buffer = processed[-1440:]


def get_latest_solar_wind() -> Optional[Dict[str, Any]]:
    """Return the most recent merged reading."""
    if not _solar_wind_buffer:
        return None
    return _solar_wind_buffer[-1]


def get_recent_readings(hours: int = 24) -> List[Dict[str, Any]]:
    """Return window of readings. In stub mode, we just return the buffer."""
    if not _solar_wind_buffer:
        # Fallback: generate mock data if NOAA is down or buffer empty
        return _generate_mock_window(hours)
    return _solar_wind_buffer


def _generate_mock_window(hours: int) -> List[Dict[str, Any]]:
    """Generate mock solar wind window for demo if NOAA is unreachable."""
    import numpy as np
    now = datetime.now(timezone.utc)
    readings = []
    for i in range(hours * 60):
        ts = (now - timedelta(minutes=i)).isoformat()
        readings.append({
            "timestamp": ts,
            "bx_gsm": float(np.random.normal(0, 2)),
            "by_gsm": float(np.random.normal(0, 2)),
            "bz_gsm": float(np.random.normal(-5, 5)), # Stormy bias
            "bt": 5.0,
            "speed": float(np.random.normal(450, 50)),
            "density": float(np.random.normal(5, 2)),
            "temperature": 100000.0,
            "kp_value": float(np.random.uniform(2, 8)),
        })
    return readings[::-1]


async def start_polling():
    """Background loop for polling."""
    while True:
        await poll_once()
        await asyncio.sleep(60)
