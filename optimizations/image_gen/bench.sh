#!/usr/bin/env bash
# Warm per-image time of one or more sd-server builds at 1024x1024 and seed 42, 8 steps and CFG 1 unless set. Each
# round starts the builds' servers in turn, so clocks and temperature hit them alike, and sends one warm-up request
# (weights to the GPU, GEMM kernel tuning) and then the three test prompts, each new to the server (it reuses the text
# encoding of a repeated prompt). Prints the median times (the whole request, prompt in and PNG out; the server's
# generate_image; its sampling steps) and each image's md5: the same hashes from two builds mean the same pixels (a
# stock build encodes its PNGs differently, so its hashes always differ).
#
#   ./bench.sh LABEL=COMMAND ...
#   ./bench.sh ours=build/bin/sd-server stock=stock/bin/sd-server
#   ./bench.sh ours=build/bin/sd-server "no-own-gemm=env GGML_CUDA_I8_GEMM=0 build/bin/sd-server"
#
# Environment: MODELS (default ./models), DIFFUSION_MODEL (default the int8 file in MODELS), STEPS (default 8), CFG
# (default 1), ROUNDS (default 3), PORT (default 7861), OUT (default ./bench-out).
set -euo pipefail
ulimit -c 0
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
models=${MODELS:-$here/models}
diffusion_model=${DIFFUSION_MODEL:-$models/moss_image_sft_int8_convrot.safetensors}
steps=${STEPS:-8}
cfg=${CFG:-1.0}
rounds=${ROUNDS:-3}
port=${PORT:-7861}
out=${OUT:-$here/bench-out}
mkdir -p "$out"

warmup='A watercolor of a lighthouse on a rocky coast at dusk'
names=(cat poster street)
prompts=(
    "a lovely cat holding a sign says 'moss-image.cpp'"
    'A vintage travel poster of a mountain lake at sunrise, with the word "LUCEBOX" in large bold letters at the top and "Local AI" in smaller letters at the bottom'
    'A photo of a red bicycle leaning against a stone wall in a narrow Italian street, soft morning light'
)

request() {  # prompt [png path]; prints the request's wall time in seconds
    python3 - "$port" "$1" "${2:-}" <<'PY'
import base64, json, sys, time, urllib.request
port, prompt, path = sys.argv[1:4]
body = json.dumps({"prompt": prompt, "width": 1024, "height": 1024, "seed": 42}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/sdapi/v1/txt2img", body, {"Content-Type": "application/json"})
start = time.perf_counter()
image = base64.b64decode(json.load(urllib.request.urlopen(req, timeout=900))["images"][0])
print(f"{time.perf_counter() - start:.3f}")
if path:
    open(path, "wb").write(image)
PY
}

for round in $(seq "$rounds"); do
    for spec in "$@"; do
        label=${spec%%=*}
        command=${spec#*=}
        log=$out/$label-$round.log
        # shellcheck disable=SC2086  # COMMAND may carry an env prefix
        $command --diffusion-model "$diffusion_model" \
            --llm "$models/Qwen3.5-9B-Q8_0.gguf" --vae "$models/flux2-vae.safetensors" \
            --tokenizer "$models/tokenizer.json" --listen-port "$port" \
            -H 1024 -W 1024 --steps "$steps" --cfg-scale "$cfg" > "$log" 2>&1 &
        pid=$!
        for _ in $(seq 600); do
            curl -s -o /dev/null "http://127.0.0.1:$port/" && break
            kill -0 "$pid" 2>/dev/null || { echo "$label: server exited, see $log" >&2; exit 1; }
            sleep 1
        done
        request "$warmup" > /dev/null
        for i in "${!names[@]}"; do
            request "${prompts[$i]}" "$out/$label-${names[$i]}.png"
        done > "$out/$label-$round.wall"
        kill "$pid"
        wait "$pid" 2>/dev/null || true
        echo "$label round $round: $(grep -oE 'generate_image completed in [0-9.]+' "$log" | tail -n 3 | grep -oE '[0-9.]+$' | tr '\n' ' ')s"
    done
done

median() {
    sort -n | awk '{ v[NR] = $1 } END { printf "%.2f s (min %.2f, max %.2f, n=%d)", NR % 2 ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2, v[1], v[NR], NR }'
}
echo
for spec in "$@"; do
    label=${spec%%=*}
    # each run's first request is the warm-up
    per_image=$(for f in "$out/$label"-[0-9]*.log; do grep -oE 'generate_image completed in [0-9.]+' "$f" | tail -n +2 | grep -oE '[0-9.]+$'; done | median)
    sampling=$(for f in "$out/$label"-[0-9]*.log; do grep -oE 'sampling completed, taking [0-9.]+' "$f" | tail -n +2 | grep -oE '[0-9.]+$'; done | median)
    wall=$(cat "$out/$label"-[0-9]*.wall | median)
    echo "$label: request $wall"
    echo "  generate_image $per_image, sampling $sampling"
    for name in "${names[@]}"; do
        printf '  %-7s %s\n' "$name" "$(md5sum < "$out/$label-$name.png" | cut -c1-32)"
    done
done
