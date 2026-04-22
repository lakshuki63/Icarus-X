#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ICARUS-X — One-command startup script
# Starts: (optionally PostgreSQL) + FastAPI backend + frontend server
# Usage:  chmod +x start.sh && ./start.sh
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚀 ICARUS-X — Space Weather Forecasting System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Copy .env if not present ──────────────────────────
if [ ! -f .env ]; then
    echo "📋 Creating .env from template..."
    cp .env.template .env
    echo "   ✅ .env created — edit if needed"
fi

# ── 2. Create directories ───────────────────────────────
mkdir -p data models logs

# ── 3. Install dependencies ─────────────────────────────
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

echo "📦 Installing dependencies..."
source venv/bin/activate || . venv/Scripts/activate 2>/dev/null
pip install -q -r requirements.txt

# ── 4. Start FastAPI backend ────────────────────────────
echo ""
echo "🌐 Starting FastAPI backend on port 8000..."
uvicorn m5_architect.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info &
BACKEND_PID=$!

# ── 5. Start frontend server ────────────────────────────
echo "🎨 Starting frontend on port 3000..."
python3 -m http.server 3000 --directory frontend &
FRONTEND_PID=$!

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ ICARUS-X is running!"
echo ""
echo "  🔹 Dashboard:  http://localhost:3000"
echo "  🔹 API Docs:   http://localhost:8000/docs"
echo "  🔹 WebSocket:  ws://localhost:8000/ws/live"
echo ""
echo "  Press Ctrl+C to stop all services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 6. Trap Ctrl+C to kill both ─────────────────────────
trap "echo '🛑 Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

# Wait for any process to exit
wait
