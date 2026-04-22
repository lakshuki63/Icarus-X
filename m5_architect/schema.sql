-- --------------------------------------------------------
-- ICARUS-X - PostgreSQL Schema
-- Creates all tables for solar wind data, forecasts, and alerts.
-- --------------------------------------------------------

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -- Solar Wind Observations (from NOAA DSCOVR poller) ---
CREATE TABLE IF NOT EXISTS solar_wind (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    bx_gsm          REAL,           -- nT
    by_gsm          REAL,           -- nT
    bz_gsm          REAL,           -- nT  (southward = negative = bad)
    bt              REAL,           -- nT  total field
    speed           REAL,           -- km/s  solar wind speed
    density         REAL,           -- p/cm3  proton density
    temperature     REAL,           -- K     proton temperature
    source          VARCHAR(32) DEFAULT 'NOAA_DSCOVR',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, source)
);

CREATE INDEX IF NOT EXISTS idx_sw_timestamp ON solar_wind(timestamp DESC);

-- -- Kp Index Observations -------------------------------
CREATE TABLE IF NOT EXISTS kp_index (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    kp_value        REAL NOT NULL,   -- 0.00 - 9.00
    source          VARCHAR(32) DEFAULT 'NOAA',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, source)
);

CREATE INDEX IF NOT EXISTS idx_kp_timestamp ON kp_index(timestamp DESC);

-- -- AR Feature Vectors (from M1 Visionary) --------------
CREATE TABLE IF NOT EXISTS ar_features (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    n_regions       INTEGER,
    f0 REAL, f1 REAL, f2 REAL, f3 REAL,
    f4 REAL, f5 REAL, f6 REAL, f7 REAL,
    f8 REAL, f9 REAL, f10 REAL, f11 REAL,
    is_stub         BOOLEAN DEFAULT FALSE,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ar_timestamp ON ar_features(timestamp DESC);

-- -- Forecast Runs (M2 -> M3 -> M4 pipeline output) -------
CREATE TABLE IF NOT EXISTS forecast_runs (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    run_timestamp   TIMESTAMPTZ NOT NULL,
    payload         JSONB NOT NULL,      -- full M4 output dict
    model_version   VARCHAR(64),
    is_stub         BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fr_run_ts ON forecast_runs(run_timestamp DESC);

-- -- Alerts History --------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    run_id          UUID REFERENCES forecast_runs(id),
    alert_level     VARCHAR(16) NOT NULL,  -- LOW, MEDIUM, HIGH, CRITICAL
    peak_horizon_hr INTEGER,
    peak_gic        REAL,
    message         TEXT,
    should_email    BOOLEAN DEFAULT FALSE,
    should_sms      BOOLEAN DEFAULT FALSE,
    acknowledged    BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(alert_level);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
