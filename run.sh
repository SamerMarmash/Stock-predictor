#!/usr/bin/env bash
# ============================================================================
# AI Stock Predictor - One-command launcher
# Usage: ./run.sh
# ============================================================================
set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$APP_DIR/.venv"

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║     📈 AI Stock Predictor            ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 is required. Install it from https://www.python.org or:"
    echo "   brew install python3"
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python $PY_VERSION found"

# Create virtual environment if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install/upgrade package
if [ ! -f "$VENV_DIR/.installed" ] || [ "$APP_DIR/pyproject.toml" -nt "$VENV_DIR/.installed" ]; then
    echo "→ Installing dependencies (this may take a minute on first run)..."
    pip install --quiet --upgrade pip
    pip install --quiet -e ".$APP_DIR" 2>/dev/null || pip install --quiet -e "$APP_DIR"
    pip install --quiet streamlit plotly
    touch "$VENV_DIR/.installed"
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies up to date"
fi

# Create .env if it doesn't exist
if [ ! -f "$APP_DIR/.env" ]; then
    if [ -f "$APP_DIR/.env.example" ]; then
        cp "$APP_DIR/.env.example" "$APP_DIR/.env"
        echo ""
        echo "📝 Created .env from .env.example"
        echo "   Edit .env to add your API keys for full AI predictions."
        echo "   The app works without API keys (technical analysis only)."
        echo ""
    fi
fi

echo ""
echo "🚀 Starting AI Stock Predictor..."
echo "   Dashboard will open at: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

# Launch Streamlit
cd "$APP_DIR"
exec streamlit run src/ui/app.py \
    --server.port=8501 \
    --server.headless=false \
    --browser.gatherUsageStats=false \
    --theme.base=dark \
    --theme.primaryColor="#6366f1" \
    --theme.backgroundColor="#0e1117" \
    --theme.secondaryBackgroundColor="#1a1f2e" \
    --theme.textColor="#e2e8f0"
