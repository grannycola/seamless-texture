#!/usr/bin/env bash
set -e
cd "${COMFY_DIR:-/app/ComfyUI}"
exec python3 main.py --listen "${HOST:-0.0.0.0}" --port "${PORT:-8188}"