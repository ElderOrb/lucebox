#!/usr/bin/env bash
# Warm per-image time of one or more luce_server builds at 1024x1024 and seed 42, 8 steps of er_sde on the simple
# schedule (the sampler ComfyUI runs for MOSS-Image) and CFG 1 unless set. Each round starts the builds' servers in
# turn, so clocks and temperature hit them alike; a server makes one warm-up image while it loads (GEMM tuning), and
# the three test prompts are each new to it (it reuses the text encoding of a repeated prompt). Prints the median
# times (the whole request, prompt in and PNG out; the server's own time per image, then its text, sampling and
# decode phases) and each image's md5: the same hashes from two builds mean the same pixels.
#
#   ./bench.sh LABEL=COMMAND ...
#   ./bench.sh ours=../../server/build/luce_server
#   ./bench.sh new=../../server/build/luce_server old=/path/to/other/luce_server
#
# Environment: MODELS (default ./models), STEPS (default 8), CFG (default 1), SAMPLER (default er_sde), SCHEDULER
# (default simple), ROUNDS (default 3), PORT (default 7861), OUT (default ./bench-out).
set -euo pipefail
ulimit -c 0
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
models=${MODELS:-$here/models}
steps=${STEPS:-8}
cfg=${CFG:-1.0}
sampler=${SAMPLER:-er_sde}
scheduler=${SCHEDULER:-simple}
rounds=${ROUNDS:-3}
port=${PORT:-7861}
out=${OUT:-$here/bench-out}
mkdir -p "$out"

names=(cat poster street)
prompts=(
    "a lovely cat holding a sign says 'moss-image.cpp'"
    'A vintage travel poster of a mountain lake at sunrise, with the word "LUCEBOX" in large bold letters at the top and "Local AI" in smaller letters at the bottom'
    'A photo of a red bicycle leaning against a stone wall in a narrow Italian street, soft morning light'
)

request() {  # prompt png-path; prints the request's wall time in seconds
    python3 - "$port" "$1" "$2" "$steps" "$cfg" "$sampler" "$scheduler" <<'PY'
import base64, json, sys, time, urllib.request
port, prompt, path, steps, cfg, sampler, scheduler = sys.argv[1:8]
body = json.dumps({"prompt": prompt, "width": 1024, "height": 1024, "seed": 42, "steps": int(steps),
                   "cfg_scale": float(cfg), "sampler_name": sampler, "scheduler": scheduler}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/sdapi/v1/txt2img", body, {"Content-Type": "application/json"})
start = time.perf_counter()
image = base64.b64decode(json.load(urllib.request.urlopen(req, timeout=900))["images"][0])
print(f"{time.perf_counter() - start:.3f}")
open(path, "wb").write(image)
PY
}

for round in $(seq "$rounds"); do
    for spec in "$@"; do
        label=${spec%%=*}
        command=${spec#*=}
        log=$out/$label-$round.log
        # shellcheck disable=SC2086  # COMMAND may carry an env prefix
        $command "$models/moss-image.gguf" --text-encoder "$models/Qwen3.5-9B-Q8_0.gguf" \
            --host 127.0.0.1 --port "$port" > "$log" 2>&1 &
        pid=$!
        trap 'kill "$pid" 2>/dev/null || true' EXIT  # a failed request must not leave the server on the GPU
        for _ in $(seq 600); do
            curl -s -o /dev/null "http://127.0.0.1:$port/v1/models" && break
            kill -0 "$pid" 2>/dev/null || { echo "$label: server exited, see $log" >&2; exit 1; }
            sleep 1
        done
        for i in "${!names[@]}"; do
            request "${prompts[$i]}" "$out/$label-${names[$i]}.png"
        done > "$out/$label-$round.wall"
        kill "$pid"
        wait "$pid" 2>/dev/null || true
        trap - EXIT
        echo "$label round $round: $(grep -oE 'generate_image completed in [0-9.]+' "$log" | grep -oE '[0-9.]+$' | tr '\n' ' ')s"
    done
done

median() {  # [printf format, default %.2f]
    sort -n | awk -v f="${1:-%.2f}" '{ v[NR] = $1 } END {
        printf f " (min " f ", max " f ", n=%d)", NR % 2 ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2, v[1], v[NR], NR }'
}
phase() {  # label phase: the phase's milliseconds from every image's "[moss-image] ..." line
    for f in "$out/$1"-[0-9]*.log; do grep -oE "$2 [0-9]+ ms" "$f" | grep -oE '[0-9]+'; done | median %.0f
}
echo
for spec in "$@"; do
    label=${spec%%=*}
    per_image=$(for f in "$out/$label"-[0-9]*.log; do grep -oE 'generate_image completed in [0-9.]+' "$f" | grep -oE '[0-9.]+$'; done | median)
    echo "$label: request $(cat "$out/$label"-[0-9]*.wall | median) s"
    echo "  generate_image $per_image s; text $(phase "$label" text) ms, sampling $(phase "$label" sampling) ms, decode $(phase "$label" decode) ms"
    for name in "${names[@]}"; do
        printf '  %-7s %s\n' "$name" "$(md5sum < "$out/$label-$name.png" | cut -c1-32)"
    done
done
