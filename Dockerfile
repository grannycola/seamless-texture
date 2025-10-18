FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git curl ca-certificates ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
 && python3 -m pip install -r /app/ComfyUI/requirements.txt

RUN git clone --depth=1 https://github.com/spinagon/ComfyUI-seamless-tiling.git /app/ComfyUI/custom_nodes/ComfyUI-seamless-tiling \
 && git clone --depth=1 https://github.com/FlyingFireCo/tiled_ksampler.git /app/ComfyUI/custom_nodes/tiled_ksampler \
 && git clone --depth=1 https://github.com/WASasquatch/was-node-suite-comfyui /app/ComfyUI/custom_nodes/was-node-suite-comfyui \
 && for req in /app/ComfyUI/custom_nodes/*/requirements.txt; do \
        [ -f "$req" ] && python3 -m pip install -r "$req" || true; \
    done

RUN set -eux; \
mkdir -p /app/ComfyUI/models/checkpoints; \
curl -fL --retry 5 --retry-delay 2 -C - \
    -o /app/ComfyUI/models/checkpoints/v2-1_768-ema-pruned.safetensors \
    "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors"

COPY --chmod=755 entrypoint.sh /entrypoint.sh

EXPOSE 8188
WORKDIR /app/ComfyUI
ENTRYPOINT ["/entrypoint.sh"]
