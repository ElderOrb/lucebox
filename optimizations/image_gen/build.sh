#!/usr/bin/env bash
# Builds sd-server and sd-cli with the image generation kernels: stable-diffusion.cpp and its ggml at the pinned
# commits, patches/ applied and ggml-cuda/ copied into ggml, a HIP build for one GPU architecture.
#
#   ./build.sh                  build into ./build, binaries in build/bin
#   ./build.sh DIR              build into DIR
#   ./build.sh --stock DIR      the same commits without our changes, for A/B runs
#
# Environment: LUCE_HIP_ARCH (default gfx1201, the Radeon AI PRO R9700), ROCM_PATH (default /opt/rocm), JOBS (default
# all cores). Needs git, cmake >= 3.25, ninja and ROCm 7.2 with the hipBLAS, hipBLASLt and rocBLAS development packages.
set -euo pipefail

SD_REPO=https://github.com/leejet/stable-diffusion.cpp
SD_COMMIT=b13a48219d5cdf408edcb93da6cd69b612ed03d3    # MOSS-Image support
GGML_COMMIT=dc998771e700d49bbdf9a39cede22bb5b3311767  # its ggml submodule

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stock=0
if [ "${1:-}" = --stock ]; then
    stock=1
    shift
fi
out=${1:-$here/build}
mkdir -p "$out"
out=$(cd "$out" && pwd)
arch=${LUCE_HIP_ARCH:-gfx1201}
rocm=${ROCM_PATH:-/opt/rocm}
src=$out/stable-diffusion.cpp

if [ ! -d "$src/.git" ]; then
    git clone --filter=blob:none "$SD_REPO" "$src"
fi
git -C "$src" checkout --quiet --force "$SD_COMMIT" 2>/dev/null ||
    { git -C "$src" fetch --quiet origin && git -C "$src" checkout --quiet --force "$SD_COMMIT"; }
git -C "$src" submodule update --init --recursive --force
if [ "$(git -C "$src/ggml" rev-parse HEAD)" != "$GGML_COMMIT" ]; then
    echo "ggml is at $(git -C "$src/ggml" rev-parse HEAD), expected $GGML_COMMIT" >&2
    exit 1
fi

# Start from the pinned sources on every run, so a rebuild never applies the patches twice.
for repo in "$src" "$src/ggml"; do
    git -C "$repo" reset --quiet --hard
    git -C "$repo" clean --quiet -fd
done
if [ "$stock" = 0 ]; then
    git -C "$src" apply "$here/patches/stable-diffusion.cpp.patch"
    git -C "$src/ggml" apply "$here/patches/ggml.patch"
    cp "$here"/ggml-cuda/*.cu "$here"/ggml-cuda/*.cuh "$src/ggml/src/ggml-cuda/"
fi

cmake -S "$src" -B "$out/cmake" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$rocm/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$rocm/llvm/bin/clang++" \
    -DSD_HIPBLAS=ON -DGPU_TARGETS="$arch" -DAMDGPU_TARGETS="$arch" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="$rocm/llvm/lib" -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build "$out/cmake" --target sd-server sd-cli --parallel "${JOBS:-$(nproc)}"
mkdir -p "$out/bin"
cp "$out/cmake/bin/sd-server" "$out/cmake/bin/sd-cli" "$out/bin/"
echo "built $out/bin/sd-server and $out/bin/sd-cli for $arch$([ "$stock" = 1 ] && echo ' (stock)' || true)"
