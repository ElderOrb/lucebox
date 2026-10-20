<p align="left">
  <a href="../../README.md">← lucebox</a>
</p>

<p align="center">
  <img src="hero.png" width="760" alt="Two 1024x1024 MOSS-Image outputs from this build at 8 steps of er_sde and CFG 2.5: a vintage travel poster with the word LUCEBOX over a mountain lake, and a red bicycle against a stone wall in a narrow street" />
</p>

<h1 align="center">MOSS-Image on one R9700</h1>

<p align="center">
  <strong>A 1024×1024 image in 3.2 s on one Radeon AI PRO R9700, from the full 5B model.</strong><br/>
  MOSS-Image SFT, 8 steps of er_sde, int8 convrot weights, no distillation and no step caching; 5.9 s at CFG 2.5, as in the pictures above.<br/>
  Our own int8 GEMMs, attention and VAE kernels for RDNA4, closer to the bf16 model than stock stable-diffusion.cpp.<br/>
  <strong>1.4×</strong> a DGX Spark running ComfyUI on the same diffusion weights and sampler, <strong>1.8×</strong> on a new prompt.
</p>

---

```
MOSS-Image SFT · 1024×1024 · 8 steps of er_sde · CFG 1 · warm server · median per image

  R9700, this build                       ██████                            3.16 s
  DGX Spark, ComfyUI, same prompt again   █████████                         4.54 s
  DGX Spark, ComfyUI, new prompt          ███████████                       5.60 s
  R9700, stable-diffusion.cpp             ████████████████████████         12.19 s
  DGX Spark, stable-diffusion.cpp, Euler  ████████████████████████████████  16.1 s

  Same int8 convrot format, steps, CFG, size and seed in every row, and the same sampler
  except in the last row. The stable-diffusion.cpp rows run the upstream commit this build
  patches. Method in RESULTS.md.
```

MOSS-Image is a 5B single-stream diffusion transformer (42 layers, 24 heads of 128) with a Qwen3.5 9B text encoder and the FLUX.2 VAE. This folder builds [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) at a pinned commit with our ggml kernels and serves it over HTTP. The model is untouched: same weights, steps, sampler and seed as stock, with nothing distilled, cached or skipped. The speed comes from the kernels.

## Quick start

Needs ROCm 7.2 with the hipBLAS, hipBLASLt and rocBLAS development packages, cmake, ninja, and python3 with numpy.

```bash
./fetch_models.sh   # pinned Hugging Face revisions into ./models (26 GB), then writes the int8 model
./build.sh          # stable-diffusion.cpp + these kernels, HIP for gfx1201, binaries in build/bin
./serve.sh          # 1024x1024, 8 steps of er_sde, CFG 1, on http://127.0.0.1:7860
```

```bash
curl -s http://127.0.0.1:7860/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a lovely cat holding a sign that says \"hello\"", "size": "1024x1024"}' |
  python3 -c 'import base64, json, sys; open("cat.png", "wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

The server also answers the AUTOMATIC1111 API (`POST /sdapi/v1/txt2img`). Fields sent with a request override the defaults of `serve.sh`.

CFG 1 is the fast setting, and at 8 steps some pictures come out misshapen: by eye, 4 of 8 in a check of two prompts with four seeds each. At CFG 2.5 (`./serve.sh --cfg-scale 2.5`) 1 of 8 did, and an image takes 5.9 s, as each step runs the model twice. Euler, stable-diffusion.cpp's default sampler, leaves 8-step pictures hazy, so `serve.sh` and `bench.sh` sample with er_sde on the simple schedule, as ComfyUI does for this model.

`./bench.sh` times builds against each other on the same card, for example `./bench.sh ours=build/bin/sd-server stock=stock/bin/sd-server` after `./build.sh --stock stock`.

## How it works

Each of the 42 transformer layers does its work in 9 kernels, plus two tiny multiplies that fold the step's modulation into the norm weights: 11 launches where stock stable-diffusion.cpp makes 36. Per denoising step that is 537 kernel launches instead of 1,594. It is not a megakernel (one launch for a whole model, as in our [Qwen 3.5 0.8B megakernel](../megakernel/README.md)), and one would gain little: at about 4,100 tokens each kernel keeps the whole GPU busy for 0.1 to 2 ms, and a kernel trace puts the layers' kernels at 315 ms of a 333 ms step, with 6 ms of gaps between them. The fusions pay because the int32 GEMM sums and the f32 intermediates mostly stay out of memory.

| Kernel (per layer, ~4,100 tokens) | Does the work of | ms |
|---|---|---:|
| int8 qkv GEMM + attention prologue | GEMM, dequantize, q and k RMS norm and RoPE, v cast to f16 | 1.27 |
| flash attention for RDNA4 | ggml's tile kernel, which reaches ~12 TFLOPS here | 1.98 |
| quantize the attention output | | 0.11 |
| int8 o GEMM + gated residual | GEMM, dequantize, gate multiply, residual add | 0.38 |
| norm + quantize | RMS norm, modulation scale and shift, quantize | 0.12 |
| int8 up/gate GEMM (hipBLASLt) | | 1.97 |
| SiTU gate + quantize | dequantize, two tanh caps, sigmoid, gate, quantize | 0.52 |
| int8 down GEMM + gated residual | GEMM, dequantize, gate multiply, residual add | 0.95 |
| norm + quantize, next layer | RMS norm, modulation scale and shift, quantize | 0.16 |

- **Own int8 GEMM** (`ggml-cuda/gemm-i8.cu`). 256×128×64 tiles on RDNA4 WMMA with the epilogue in registers, so the int32 sums never go to memory. Keeping math out of the GEMMs mattered as much as fusing: the card runs at its power limit, so the SiTU gate stays a separate memory-bound pass, where it costs less than inside the GEMM.
- **Attention** (`ggml-cuda/fattn-rdna4.cu`). Head size 128 without a mask, one query per lane, the next K/V tile prefetched while the current one computes, partial query tiles scheduled last.
- **Row kernels** (`ggml-cuda/dit-i8.cu`, `convrot-wave.cuh`). Norm, modulation and the 256-wide Hadamard rotation of the int8 convrot format in one pass per row, with the row in registers.
- **Graph** (`patches/stable-diffusion.cpp.patch`). The FFN gate is one SiTU op where the stock graph builds nine and copies two slices, the timestep modulation is folded into the norm weights once per step, and q, k and v are views of one GEMM output.
- **VAE decode** (`ggml-cuda/conv2d-rdna4.cu`, `group-norm-fused.cu`). An implicit-GEMM convolution for RDNA4 that reads the nearest-upscale source and adds the bias and skip connection in its epilogue, fed by a group norm and SiLU that write its f16 input directly. The decode takes 0.19 s, stock 0.84 s.
- **int8 GEMM tuning** (`ggml-cuda/hipblaslt-i8.cu`). hipBLASLt picks from 16 timed candidates once per shape and per 256 tokens, so a new prompt length does not re-tune every request.

## Correctness

- **The own GEMMs are exact.** int8 sums are exact integers, and the fused epilogues repeat the dequantize, gate and residual arithmetic of the separate kernels in the same order: with `GGML_CUDA_I8_GEMM=0` the three test prompts give byte-identical PNGs.
- **Closer to the bf16 model than stock.** Against the bf16 release weights run with f32 attention, this build's images score a PSNR of 47.3 dB after one step and 35.2 dB after eight; stock's int8 path scores 39.8 and 27.6 dB (mean of the three test prompts, sampled with Euler). Stock's error comes mostly from its attention path: its own bf16 run scores 26.8 dB after eight steps.
- The weights are the int8 convrot format ComfyUI ships for this model: per-row int8 weights rotated in 256-wide Hadamard groups, activations quantized per token at run time. `convert_int8_convrot.py` writes them from the bf16 release with numpy only, and the result does not depend on the machine.

## Switches

| Variable | Effect |
|---|---|
| `GGML_CUDA_I8_GEMM=0` | int8 linears on hipBLASLt with separate epilogue kernels: same pixels, 3.3 s per image |
| `GGML_CUDA_DISABLE_FUSION=1` | ggml's own switch, every fusion off, ours and ggml's: 5.1 s per image, same accuracy, not byte-identical |
| `LUCE_HIP_ARCH` | build target for `build.sh`, default `gfx1201` |

## Limits

- Measured on the Radeon AI PRO R9700 (RDNA4, gfx1201) only. The GEMM, attention and convolution kernels run on RDNA4 and the row kernels on RDNA2 and newer; elsewhere ggml's own kernels run, untested.
- The attention kernel takes head size 128 without a mask, as in MOSS-Image.
- The patches apply to stable-diffusion.cpp `b13a482` and its ggml `dc99877`; `build.sh` checks both out.
- The card ran at its 330 W power limit, the driver's maximum. At the 300 W default an image takes about 1% longer.

## Files

| Path | Contents |
|---|---|
| `ggml-cuda/` | the new kernels, copied into ggml at build time |
| `patches/ggml.patch` | ggml's own files: the SiTU op, fusion matching, the hipBLASLt int8 path, RDNA4 attention selection, small fixes |
| `patches/stable-diffusion.cpp.patch` | the MOSS-Image graph, f16 fallback attention for the VAE, PNG at zlib level 1 |
| `build.sh`, `serve.sh`, `bench.sh` | build, serve, A/B timing |
| `fetch_models.sh`, `convert_int8_convrot.py` | weights at pinned revisions, the int8 conversion |

Numbers and method: [RESULTS.md](RESULTS.md).
