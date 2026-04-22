"""
ICARUS-X — Database Connection Manager (M5 Architect)

Provides async SQLAlchemy engine + session factory for PostgreSQL.
Falls back to SQLite if PostgreSQL is unavailable (demo mode).

Inputs:  DATABASE_URL from environment
Outputs: get_session() async context manager, engine instance
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, text
from loguru import logger

# ── Load .env if present ─────────────────────────────────
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    _template = Path(__file__).resolve().parent.parent / ".env.template"
    if _template.exists():
        load_dotenv(_template)
        logger.warning("[!] Using .env.template -- copy to .env for production")

# ── Configuration ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SQLITE_PATH = _PROJECT_ROOT / "data" / "icarus_x.db"

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    f"sqlite+aiosqlite:///{_SQLITE_PATH}",
)
DATABASE_URL_SYNC: str = os.environ.get(
    "DATABASE_URL_SYNC",
    f"sqlite:///{_SQLITE_PATH}",
)

# ── Determine if using SQLite (demo) or PostgreSQL ───────
_is_sqlite = "sqlite" in DATABASE_URL.lower()

if _is_sqlite:
    logger.info(f"[DB] Using SQLite database at {_SQLITE_PATH}")
    _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
else:
    logger.info("[DB] Using PostgreSQL database")

# ── Async Engine ─────────────────────────────────────────
try:
    async_engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        **({"pool_size": 5, "max_overflow": 10} if not _is_sqlite else {}),
    )
    logger.info("[OK] Async database engine created")
except Exception as e:
    logger.error(f"[ERR] Failed to create async engine: {e}")
    # Fallback to SQLite
    DATABASE_URL = f"sqlite+aiosqlite:///{_SQLITE_PATH}"
    _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    async_engine = create_async_engine(DATABASE_URL, echo=False)
    logger.warning("[!] Fell back to SQLite")

# ── Session Factory ──────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session with automatic commit/rollback."""
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db() -> None:
    """Create all tables using raw SQL from schema.sql or inline DDL."""
    schema_path = Path(__file__).resolve().parent / "schema.sql"

    if _is_sqlite:
        # Use simplified SQLite-compatible DDL
        await _init_sqlite()
        return

    if schema_path.exists():
        sql = schema_path.read_text(encoding="utf-8")
        async with async_engine.begin() as conn:
            # Split on semicolons and execute each statement
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if stmt and not stmt.startswith("--"):
                    try:
                        await conn.execute(text(stmt))
                    except Exception as e:
                        logger.warning(f"Schema statement skipped: {e}")
        logger.info("[OK] PostgreSQL schema initialized")
    else:
        logger.warning("[!] schema.sql not found -- tables may not exist")


async def _init_sqlite() -> None:
    """Create SQLite-compatible tables for demo mode."""
    ddl = """
    CREATE TABLE IF NOT EXISTS solar_wind (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        bx_gsm REAL, by_gsm REAL, bz_gsm REAL, bt REAL,
        speed REAL, density REAL, temperature REAL,
        source TEXT DEFAULT 'NOAA_DSCOVR',
        ingested_at TEXT DEFAULT (datetime('now')),
        UNIQUE(timestamp, source)
    );
    CREATE TABLE IF NOT EXISTS kp_index (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        kp_value REAL NOT NULL,
        source TEXT DEFAULT 'NOAA',
        ingested_at TEXT DEFAULT (datetime('now')),
        UNIQUE(timestamp, source)
    );
    CREATE TABLE IF NOT EXISTS ar_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL UNIQUE,
        n_regions INTEGER,
        f0 REAL, f1 REAL, f2 REAL, f3 REAL,
        f4 REAL, f5 REAL, f6 REAL, f7 REAL,
        f8 REAL, f9 REAL, f10 REAL, f11 REAL,
        is_stub INTEGER DEFAULT 0,
        ingested_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS forecast_runs (
        id TEXT PRIMARY KEY,
        run_timestamp TEXT NOT NULL,
        payload TEXT NOT NULL,
        model_version TEXT,
        is_stub INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS alerts (
        id TEXT PRIMARY KEY,
        run_id TEXT,
        alert_level TEXT NOT NULL,
        peak_horizon_hr INTEGER,
        peak_gic REAL,
        message TEXT,
        should_email INTEGER DEFAULT 0,
        should_sms INTEGER DEFAULT 0,
        acknowledged INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now'))
    );
    """
    async with async_engine.begin() as conn:
        for stmt in ddl.split(";"):
            stmt = stmt.strip()
            if stmt:
                await conn.execute(text(stmt))
    logger.info("[OK] SQLite schema initialized (demo mode)")


def get_sync_engine():
    """Get a synchronous engine for training scripts that need sync DB access."""
    return create_engine(DATABASE_URL_SYNC, echo=False)
