#!/usr/bin/env bash
# One-time setup on a Mac with Apple Silicon (M1/M2/M3).
# Makes a native arm64 Python environment in .venv, installs the packages,
# runs the tests and records the software versions.
#   bash setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
ARCH=$($PY -c 'import platform; print(platform.machine())')
if [ "$ARCH" != "arm64" ]; then
  echo "WARNING: this Python runs as $ARCH, not arm64 (probably Rosetta)."
  echo "         Install an arm64 Python (e.g. Miniforge or python.org) and run again,"
  echo "         or set PYTHON=/path/to/arm64/python3 bash setup_mac.sh"
fi

$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# opencv-python (not -headless): the review window and the mask editor need a GUI.
pip install -r requirements.txt

python - <<'PY'
import platform, sys
print("python", sys.version.split()[0], "on", platform.machine())
assert platform.machine() == "arm64", "not a native arm64 Python"
PY

# Nothing local needs PyTorch (SAM 3 runs on Modal's GPU). If you add local PyTorch work,
# use the Apple GPU like this:  export PYTORCH_ENABLE_MPS_FALLBACK=1   and   device = "mps".

python -m pytest -q tests
python scripts/env_report.py
echo
echo "Setup done. Next time just run:  source .venv/bin/activate"
echo "Modal login (once):              modal token new"
