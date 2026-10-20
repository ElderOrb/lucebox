# MOSS-Image results

**Setup.** AMD Radeon AI PRO R9700 (RDNA4 `gfx1201`, 32 GB) at its 330 W power limit, ROCm 7.2.4. stable-diffusion.cpp `b13a482` with ggml `dc99877`, patched as in this folder or stock. MOSS-Image SFT as int8 convrot (`convert_int8_convrot.py` from the bf16 release), Qwen3.5 9B Q8_0 text encoder, FLUX.2 VAE. 1024×1024, 8 steps, CFG 1, Euler, seed 42.

**Method.** `bench.sh`: three rounds; in each, every build's `sd-server` starts, takes one warm-up request, then the three test prompts, each new to the server. Builds alternate inside a round, so clocks and temperature hit them alike. Medians of 9 images. *Request* is timed by the client, prompt in and PNG out over HTTP; *generate_image* is the server's text encoding, sampling and VAE decode.

## Per image on the R9700

| Build | Request | generate_image | Sampling, 8 steps |
|---|---:|---:|---:|
| **This build** | **3.00 s** | 2.96 s | 2.72 s |
| `GGML_CUDA_I8_GEMM=0` (hipBLASLt + separate epilogues) | 3.20 s | 3.16 s | 2.92 s |
| `GGML_CUDA_DISABLE_FUSION=1` (no fused kernels) | 4.92 s | 4.88 s | 4.47 s |
| stable-diffusion.cpp `b13a482`, stock | 11.83 s | 11.71 s | 10.83 s |

This build's request splits into text encoding 0.03 s, sampling 2.72 s, VAE decode 0.19 s, and 0.04 s for the PNG and HTTP. Stock decodes the VAE in 0.84 s. The first request after a server start takes 8.4 s here and 16.8 s on stock: weights move to the GPU, kernels load, and hipBLASLt tunes its GEMM.

At the model card's settings, 50 steps with CFG 3 (two forwards a step), one round of three images: this build 33.9 s per image, stock 134.7 s.

## Accuracy

The reference is the bf16 release weights on stock stable-diffusion.cpp with f32 attention everywhere (`--flash-attn off`), the most exact run the code offers. PSNR of each build's images against it, higher is closer; one step shows a single forward pass's error, eight steps show it compounded:

| Build | Weights | Steps | cat | poster | street | Mean |
|---|---|---:|---:|---:|---:|---:|
| stock | int8 | 1 | 40.56 | 39.13 | 39.73 | 39.8 |
| **this build** | int8 | 1 | 44.50 | 46.06 | 51.30 | **47.3** |
| stock | bf16 | 8 | 21.81 | 27.12 | 31.54 | 26.8 |
| stock | int8 | 8 | 24.62 | 28.14 | 29.99 | 27.6 |
| **this build** | int8 | 8 | 33.92 | 35.26 | 36.47 | **35.2** |
| `GGML_CUDA_DISABLE_FUSION=1` | int8 | 8 | 32.93 | 32.86 | 38.48 | 34.8 |

- This build lands closer to the reference than stock does. Stock's own bf16 run with its default flash attention lands no closer than its int8 run, so on this card most of stock's error comes from its attention path, not from the int8 weights.
- `GGML_CUDA_I8_GEMM=0` writes byte-identical PNGs to this build (md5 `a1604a02`, `3e4b63f9`, `8b8599c0` for cat, poster, street).
- `GGML_CUDA_DISABLE_FUSION=1` turns off every fusion, ours and ggml's. Some round differently from the separate ops (our fused group norm among them), so its images are not byte-identical to this build's (PSNR 35-43 dB between the two), but they are as close to the reference.
- This build and stock differ from each other by PSNR 25-33 dB after eight steps. Composition and lettering agree; fine detail differs, as it does between any two of the runs above.

## Against a DGX Spark

The same prompts, seed, size, steps and CFG on a DGX Spark (GB10), warm:

| DGX Spark setup | Prompt repeated | New prompt | This build is |
|---|---:|---:|---:|
| ComfyUI, int8 convrot diffusion weights from our converter, 16-bit attention, full VAE | 4.54 s | 5.60 s | **1.5× / 1.9×** |
| ComfyUI, its shipped files: int8 attention, small VAE | 4.22 s | 5.34 s | 1.4× / 1.8× |
| stable-diffusion.cpp `b13a482` (CUDA), same int8 weights | | 16.1 s | 5.4× |

- ComfyUI 0.42.0 (`b1a247c`), PyTorch 2.14.1 with CUDA 13.0, comfy-kitchen 0.2.39, its native MOSS-Image nodes, its default sampler for the model (er_sde, simple schedule; one model call per step, like Euler). Its text encoder is the shipped Qwen3.5 9B w4a8 in both ComfyUI rows; ours is Q8_0.
- Times are ComfyUI's own "Prompt executed in", which includes saving the PNG but not the HTTP round trip. ComfyUI keeps the previous request's text encoding, so the same prompt twice in a row skips it: one such request per setup, against five new-prompt requests (5.59-5.61 s).
- The R9700 side is this build's request time, so the comparison counts more work on our side, not less.
- ComfyUI per new-prompt image: text encoding 1.07 s, sampling 3.95 s (0.49 s a step), VAE 0.45 s, saving 0.12-0.29 s. Ours: 0.03 s, 2.72 s (0.34 s a step), 0.19 s and 0.04 s.
- The shipped files trade quality for speed: in our poster test they garbled the word "LUCEBOX", which the 16-bit attention run kept readable (one prompt, one seed).

## Seven calls

stable-diffusion.cpp's discrete schedule spreads the steps evenly over timesteps 999 down to 0, and with this model's flow shift of 5, timestep 0 is noise level 0.005. The 8 calls run at:

| Call | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Noise level | 1.000 | 0.968 | 0.926 | 0.870 | 0.790 | 0.667 | 0.456 | 0.005 |

The seventh call's Euler step lands at noise level 0.005, and the eighth call removes that last bit. Passing the schedule without its 0.005 point (`--sigmas 1,0.967778325,0.926021934,0.869763613,0.789861202,0.667443454,0.456277341,0`) steps from 0.456 straight to the clean image after the same seven calls. Passed whole the same way, the schedule gives images byte-identical to the default (20 of 20), so the seven calls are exactly the default's first seven.

20 prompts used nowhere else in this folder, seeds 42 and 7, one warm server per column. Times are medians and PSNRs means over the 40 images:

| | 8 calls | 7 calls |
|---|---:|---:|
| Request | 3.02 s | 2.67 s |
| Sampling | 2.76 s | 2.40 s |
| PSNR against the 8-call image | | 59.3 dB (54.0 to 65.6) |
| PSNR against the bf16 reference | 36.66 dB | 36.66 dB |

- 92% of pixel values are identical between the two (79% in the least similar image). The median image has 4 pixels more than 4 levels apart out of 255. The least similar one, a pen-drawn map, has 277, at most 23 levels apart, and the two versions look the same at 4× zoom.
- Per image, the 7-call version is between 0.28 dB farther from the reference and 0.11 dB closer than the 8-call one, 0.00 dB on average.
- `bench.sh` on the three test prompts agrees: 3.02 s and 2.67 s per request.
- The comparisons in this file keep all 8 calls. ComfyUI's simple schedule for this model has no such point: its 8 calls run from noise level 1.0 down to 0.42, so its last call still has work to do.

## Where a step goes

Per transformer layer at 4,116 tokens (4,096 for the image plus the cat prompt's 20), from a rocprofv3 kernel trace of an 8-step `sd-cli` run: medians over the 217 layers of steps 2 to 7 with consistent timestamps.

| Kernel | µs |
|---|---:|
| int8 qkv GEMM + attention prologue | 1,274 |
| flash attention | 1,984 |
| quantize the attention output | 113 |
| int8 o GEMM + gated residual | 380 |
| norm + quantize | 120 |
| int8 up/gate GEMM (hipBLASLt) | 1,972 |
| SiTU gate + quantize | 518 |
| int8 down GEMM + gated residual | 953 |
| norm + quantize, next layer | 160 |
| four padding memsets and two copies | 27 |
| gaps between those 15 operations | 145 |
| **layer** | **7,646** |

In the trace a step takes 333 ms: 315 ms of the layers' kernels, 6 ms of gaps between them, and 12 ms outside the layers (the timestep and output layers, the sampler, the next step's inputs). The int8 GEMMs run at 180-220 TOPS with their epilogues and the attention at about 105 TFLOPS. The card sits at its power limit through a request, so the GEMMs clock near 2.2 GHz instead of the 2.9 GHz behind the spec sheet's 383 int8 TOPS.

## Kernel launches

Counted from the HIP runtime's API log (`AMD_LOG_LEVEL=3`): a 2-step `sd-cli` run minus a 1-step run leaves one denoising step.

| Per denoising step | Stock | This build |
|---|---:|---:|
| kernel launches | 1,594 | 537 |
| memsets and copies | 649 | 277 |
| kernel launches per transformer layer | 36 | 11 |

Per layer this build launches the 9 kernels above and two vector multiplies that fold the step's modulation into the norm weights, plus four one-row memsets for the int8 padding, a copy of the qkv input (the allocator hands its buffer to k) and a copy of the activation scales. Stock spends its 36 on separate GEMMs, dequantize and quantize passes, norms, RoPE, casts and the elementwise ops of the modulation and the gate.

It is not a megakernel, which would run a whole step in one launch, and one would gain little here: at about 4,100 tokens every kernel fills the GPU for 0.1 to 2 ms, so the gaps between the layers' kernels come to 6 ms of a 333 ms step. Even one launch for the whole run, sampler included, could only save those 6 ms and part of the 12 ms outside the layers, about 5% at the very most, and it would have to replace hipBLASLt's up/gate GEMM and give every stage one register and LDS budget. The fusions pay by keeping int32 sums and f32 intermediates out of memory, not by saving launches.

## Power

The R9700 runs at its power limit during every request. These numbers use 330 W, the most the driver allows; the card ships at 300 W, which makes an image about 1% slower. Images are bit-identical at either limit.
