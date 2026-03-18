#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="/Applications/Python_AI/Neural_Image_Caption_Generation"
PYTHON_BIN="/opt/anaconda3/envs/tf-metal/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Khong tim thay Python env tf-metal tai: $PYTHON_BIN"
  exit 1
fi

export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

exec "$PYTHON_BIN" -m streamlit run "$PROJECT_ROOT/streamlit_app.py"
