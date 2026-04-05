#!/usr/bin/env bash
set -euo pipefail
python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
python3 -m pip install --no-cache-dir \
  "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" \
  --index-url https://download.pytorch.org/whl/cu121
