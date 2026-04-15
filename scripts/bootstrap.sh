#!/usr/bin/env bash
# bootstrap.sh
# ------------
# One-command environment setup for the Congestion Pricing Impact Analyzer.
# Creates a conda environment, installs all dependencies, and validates imports.
#
# Usage:
#   chmod +x scripts/bootstrap.sh
#   ./scripts/bootstrap.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="mobility_policy"
PYTHON_VERSION="3.11"

echo "============================================================"
echo "  Congestion Pricing Impact Analyzer — Bootstrap"
echo "============================================================"
echo ""

# ── 1. Check conda ────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
  echo "[ERROR] conda not found. Please install Miniconda or Anaconda."
  echo "        https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

# ── 2. Create environment ─────────────────────────────────────────────────────
echo "[1/5] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "      Environment already exists. Skipping creation."
else
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

# ── 3. Activate ───────────────────────────────────────────────────────────────
echo "[2/5] Activating environment..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── 4. Install dependencies ───────────────────────────────────────────────────
echo "[3/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r "${PROJECT_ROOT}/requirements.txt"

# Install GDAL system dependency if on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  if command -v apt-get &>/dev/null; then
    echo "      Installing GDAL system deps..."
    sudo apt-get install -y libgdal-dev libgeos-dev libproj-dev 2>/dev/null || true
  fi
fi

# ── 5. Create artifact directories ───────────────────────────────────────────
echo "[4/5] Creating artifact directories..."
mkdir -p "${PROJECT_ROOT}/artifacts"/{raw/{yellow,green,fhvhv,auxiliary},processed/{yellow,green,fhvhv},results,figures,tables}

# ── 6. Validate imports ───────────────────────────────────────────────────────
echo "[5/5] Validating core imports..."
python - <<'PYEOF'
import sys
failed = []
modules = [
    "pandas", "numpy", "pyarrow", "duckdb",
    "geopandas", "statsmodels", "linearmodels",
    "sklearn", "scipy", "plotly", "streamlit",
    "loguru", "yaml", "requests", "tqdm",
]
for m in modules:
    try:
        __import__(m)
        print(f"  ✓ {m}")
    except ImportError as e:
        print(f"  ✗ {m}: {e}")
        failed.append(m)

optional = ["econml", "doubleml", "pysyncon"]
for m in optional:
    try:
        __import__(m)
        print(f"  ✓ {m} (optional)")
    except ImportError:
        print(f"  ~ {m} not installed (optional, fallback available)")

if failed:
    print(f"\n[WARNING] {len(failed)} required module(s) failed: {failed}")
    sys.exit(1)
else:
    print("\n✅ All required dependencies verified.")
PYEOF

echo ""
echo "============================================================"
echo "  Bootstrap complete!"
echo ""
echo "  Activate environment:  conda activate ${ENV_NAME}"
echo "  Download data:         python src/data/download_tlc.py"
echo "  Run full pipeline:     python src/run_pipeline.py"
echo "  Skip download:         python src/run_pipeline.py --skip-download"
echo "  Launch dashboard:      streamlit run src/dashboard/app.py"
echo "============================================================"
