#!/usr/bin/env bash
# Serves MOSS-Image from ./build with the measured settings: 1024x1024, 8 steps, CFG 1, and the er_sde sampler on the
# simple schedule that ComfyUI runs for this model (Euler, stable-diffusion.cpp's default, leaves 8-step pictures
# hazy). sd-server answers the OpenAI images API (POST /v1/images/generations) and the AUTOMATIC1111 one
# (POST /sdapi/v1/txt2img), and fields sent with a request override these defaults. Extra arguments go to sd-server.
#
#   ./serve.sh [sd-server flags]
#
# Environment: MODELS (default ./models), HOST (default 127.0.0.1), PORT (default 7860).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
models=${MODELS:-$here/models}
exec "$here/build/bin/sd-server" \
    --diffusion-model "$models/moss_image_sft_int8_convrot.safetensors" \
    --llm "$models/Qwen3.5-9B-Q8_0.gguf" \
    --vae "$models/flux2-vae.safetensors" \
    --tokenizer "$models/tokenizer.json" \
    --listen-ip "${HOST:-127.0.0.1}" --listen-port "${PORT:-7860}" \
    -H 1024 -W 1024 --steps 8 --cfg-scale 1.0 --sampling-method er_sde --scheduler simple "$@"
