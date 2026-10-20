#!/usr/bin/env bash
# Downloads the MOSS-Image SFT files at pinned revisions and writes the int8 diffusion model:
#   moss_image_sft_bf16.safetensors          the diffusion transformer, bf16 (converted, then no longer needed)
#   moss_image_sft_int8_convrot.safetensors  the same with int8 convrot linears, written by convert_int8_convrot.py
#   Qwen3.5-9B-Q8_0.gguf                     the text encoder
#   flux2-vae.safetensors                    the image decoder
#   tokenizer.json                           the text encoder's tokenizer
#
#   ./fetch_models.sh [DIR]     (default ./models: 26 GB, 15 GB once the bf16 file is deleted)
#
# Needs curl and python3 with numpy.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dir=${1:-$here/models}
mkdir -p "$dir"

get() {  # repo revision path name
    if [ -s "$dir/$4" ]; then
        echo "have $4"
        return
    fi
    echo "fetching $4"
    curl -L --fail --retry 3 -C - -o "$dir/$4.part" "https://huggingface.co/$1/resolve/$2/$3"
    mv "$dir/$4.part" "$dir/$4"
}
get Comfy-Org/MOSS-Image 3284a1dbb351e3396d70f39f2a71cbf6ed2ef1bb \
    diffusion_models/moss_image_sft_bf16.safetensors moss_image_sft_bf16.safetensors
get Comfy-Org/MOSS-Image 3284a1dbb351e3396d70f39f2a71cbf6ed2ef1bb vae/flux2-vae.safetensors flux2-vae.safetensors
get unsloth/Qwen3.5-9B-GGUF 3885219b6810b007914f3a7950a8d1b469d598a5 Qwen3.5-9B-Q8_0.gguf Qwen3.5-9B-Q8_0.gguf
get OpenMOSS-Team/MOSS-Image-SFT 2c7954437269461767f533b006823e8f85973a60 tokenizer/tokenizer.json tokenizer.json

if [ ! -s "$dir/moss_image_sft_int8_convrot.safetensors" ]; then
    python3 "$here/convert_int8_convrot.py" "$dir/moss_image_sft_bf16.safetensors" "$dir/moss_image_sft_int8_convrot.part"
    mv "$dir/moss_image_sft_int8_convrot.part" "$dir/moss_image_sft_int8_convrot.safetensors"
fi
echo "models in $dir"
