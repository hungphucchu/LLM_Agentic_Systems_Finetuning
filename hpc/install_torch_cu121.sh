#!/usr/bin/env bash
# PyPI's default torch pulls CUDA 13.x libs and fails on older cluster drivers
# ("driver too old", torch.version.cuda == 13.x). Replace with CUDA 12.1 wheels.
set -euo pipefail
python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
# Pin +cu121 so pip cannot substitute the PyPI CUDA-13 build.
python3 -m pip install --no-cache-dir \
  "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" \
  --index-url https://download.pytorch.org/whl/cu121
