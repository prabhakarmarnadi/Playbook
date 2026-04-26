#!/usr/bin/env bash
# Clustering V2 Hackathon — Setup Script
# Replaces: Docker, Azure CLI, gRPC proto compilation, ServiceBus configuration
# Run once: ./setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  Clustering V2 — Hackathon MVP Setup"
echo "  Replacing: Docker + AIDB + Azure Blob + Redis + ServiceBus"
echo "  With:      Poetry + DuckDB + local embeddings"
echo "═══════════════════════════════════════════════════════════════"

# ── 1. Check Poetry ────────────────────────────────────────────────────────────
if ! command -v poetry &> /dev/null; then
    echo "[1/4] Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
else
    echo "[1/4] Poetry found: $(poetry --version)"
fi

# ── 2. Install dependencies ────────────────────────────────────────────────────
echo "[2/4] Installing dependencies via Poetry..."
poetry install --no-interaction

# ── 3. Create data directories ─────────────────────────────────────────────────
echo "[3/4] Creating data directories..."
mkdir -p data/sample_agreements

# ── 4. Copy sample data if available ───────────────────────────────────────────
LEGACY_DATA="$SCRIPT_DIR/../legacy_stack/apr-agreement-clustering/examples/data"
if [ -d "$LEGACY_DATA" ]; then
    echo "[4/4] Linking legacy sample data..."
    if [ ! -f "data/sample_agreements/sample_documents.json" ]; then
        cp "$LEGACY_DATA/sample_documents.json" data/sample_agreements/ 2>/dev/null || true
    fi
    echo "  Legacy sample data available."
else
    echo "[4/4] No legacy sample data found at $LEGACY_DATA"
    echo "  Place .txt or .pdf files in data/sample_agreements/"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Before running, set your OpenAI API key:"
echo "    export OPENAI_API_KEY='sk-...'"
echo ""
echo "  Or use Ollama (local, no API key needed):"
echo "    export LLM_BACKEND=ollama"
echo "    export OLLAMA_MODEL=llama3.1"
echo ""
echo "  Run the app:"
echo "    poetry run streamlit run app.py --server.port 8501"
echo ""
echo "  Run the validation test:"
echo "    poetry run python scripts/validate_pipeline.py"
echo "═══════════════════════════════════════════════════════════════"
