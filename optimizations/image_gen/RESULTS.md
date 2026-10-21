# MOSS-Image results

**Setup.** AMD Radeon AI PRO R9700 (RDNA4 `gfx1201`, 32 GB) at its 330 W power limit, ROCm 7.2.4. luce_server built for `gfx1201`. MOSS-Image SFT as int8 convrot with the FLUX.2 VAE in one GGUF (`server/scripts/convert_moss_image.py` from the bf16 release), Qwen3.5 9B Q8_0 text encoder. 1024×1024, 8 steps of er_sde on the simple schedule (the sampler ComfyUI runs for this model), CFG 1, seed 42, unless a table says otherwise.

**Method.** `bench.sh`: three rounds; in each, the server starts (the first start tunes hipBLASLt with two warm-up images, later starts read the tuned kernels back), then takes the three test prompts, each new to it. Medians of 9 images. *Request* is timed by the client, prompt in and PNG out over HTTP; *generate_image* is the server's text encoding, sampling and VAE decode.

## Per image on the R9700

| Setting | Request | generate_image | Text | Sampling, 8 steps | Decode |
|---|---:|---:|---:|---:|---:|
| **CFG 1** | **3.06 s** | 3.03 s | 0.03 s | 2.82 s | 0.18 s |
| CFG 2.5 | 5.78 s | 5.74 s | 0.03 s | 5.54 s | 0.18 s |

A step takes 0.35 s at CFG 1 and 0.69 s at CFG 2.5, which runs the model twice. The PNG and the HTTP round trip add 0.03 s.

## Against the stable-diffusion.cpp build it replaces

This engine replaces a build of stable-diffusion.cpp `b13a482` with the same kernels as ggml patches, served by its `sd-server`. Both servers took the same requests in one session from one client script: per setting a warm-up request, then three rounds of the three test prompts, so from the second round each prompt repeats (both keep the text encoding of the last four prompts). Medians of 9 images:

| Setting | stable-diffusion.cpp build | luce_server |
|---|---:|---:|
| Euler, 1 step | 0.597 s | **0.541 s** |
| Euler, 8 steps | **3.028 s** | 3.040 s |
| er_sde, 8 steps | 3.109 s | **3.063 s** |
| er_sde, 8 steps, CFG 2.5 | 5.821 s | **5.766 s** |

- The model costs about the same in both: sampling took 2.79 s there and 2.84 s here with Euler, 2.86 and 2.87 s with er_sde, and 5.58 and 5.57 s at CFG 2.5. The decode takes 0.20 s there and 0.18 s here.
- Euler at 8 steps is the one setting where the old build was faster, by 12 ms: its Euler sampling was 0.05 s shorter, while its er_sde sampling took 0.07 s longer than its Euler. Here the two samplers differ by 0.03 s.
- With `bench.sh`'s method (new prompts, a fresh server each round), the old build took 3.16 s per request at CFG 1 (generate_image 3.11 s, sampling 2.87 s) and 5.92 s at CFG 2.5; stock stable-diffusion.cpp `b13a482` took 12.19 s at CFG 1.
- luce_server is ready 4.4-4.5 s after it starts, from the page cache: the model (2.9 s) and the text encoder (4.3 s) load at once while hipBLASLt loads the kernels tuned on the first start, so the first request runs at full speed. The first start tunes them with two warm-up images (3.0 s) and is ready after 7.2-7.5 s; the starts after it pick the same kernels, and their images match its bit for bit. `sd-server` was ready after 6.0 s and took 8.5 s on its first request.
- With the text encoder on the APU beside the R9700 (a Ryzen AI Max+ 395's Radeon 8060S, `--text-encoder-device hip:1`, built for `gfx1201;gfx1151`; the R9700 behind a PCIe Gen3 x4 link), luce_server is ready after 2.0-2.1 s: the model (2.0 s) has the R9700's link to itself while the encoder loads into the APU's memory (1.1 s). generate_image then takes 3.06-3.09 s for a new prompt, whose encoding takes 78-98 ms there instead of 25-28 ms, and 2.98-3.00 s for a repeated one. The APU rounds the encoder differently, so the images are close to the R9700-encoded ones, not byte-identical: PSNR 35.6, 44.0 and 37.9 dB on three prompts (seeds 71, 73 and 74), the same composition in each, with one local detail redrawn in the first.

## Accuracy

The reference is the bf16 release weights on stock stable-diffusion.cpp with f32 attention everywhere (`--flash-attn off`), the most exact run its code offers. Every run in this section samples with Euler on the discrete schedule. PSNR of each build's images against it, higher is closer; one step shows a single forward pass's error, eight steps show it compounded:

| Build | Weights | Steps | cat | poster | street | Mean |
|---|---|---:|---:|---:|---:|---:|
| stable-diffusion.cpp, stock | int8 | 1 | 40.56 | 39.13 | 39.73 | 39.8 |
| stable-diffusion.cpp build | int8 | 1 | 44.50 | 46.06 | 51.30 | 47.3 |
| **luce_server** | int8 | 1 | 51.72 | 47.90 | 52.61 | **50.7** |
| stable-diffusion.cpp, stock | bf16 | 8 | 21.81 | 27.12 | 31.54 | 26.8 |
| stable-diffusion.cpp, stock | int8 | 8 | 24.62 | 28.14 | 29.99 | 27.6 |
| stable-diffusion.cpp build | int8 | 8 | 33.92 | 35.26 | 36.47 | 35.2 |
| **luce_server** | int8 | 8 | 36.76 | 32.51 | 35.73 | **35.0** |

- After one step luce_server lands closest to the reference. After eight, the two int8 builds score the same within the spread of three images: the per-image scores move both ways (+2.8 dB for the cat, -2.8 dB for the poster).
- The two builds do not round alike everywhere (for example, the text encoder is the engine's own Qwen3.5 graph, and the VAE's attention runs in f16 where `sd-server`'s runs in f32), so their images are close, not byte-identical: PSNR 47-53 dB between them after one step, 34-45 dB after eight, and 27-29 dB with er_sde at CFG 2.5, whose noise each step compounds the differences. Composition and lettering agree.
- The initial noise is stable-diffusion.cpp's, bit for bit (its Philox generator and Box-Muller transform), and so is er_sde's noise at each step.
- Stock's own bf16 run with its default flash attention lands no closer than its int8 run, so on this card most of stock's error comes from its attention path, not from the int8 weights.

## Sampler and CFG

- At 8 steps, Euler leaves images hazy on its discrete schedule as on the simple one; er_sde does not, on either. Both make one model call a step.
- At CFG 1, some images come out misshapen. In a check by eye of two prompts (a fox in fresh snow, a fisherman mending a net) with seeds 1 to 4, 4 of the 8 images were misshapen at CFG 1 and 1 of 8 at CFG 2.5. The server defaults to CFG 1, which runs the model once a step; a request sets CFG 2.5 with `cfg_scale` or `guidance.txt_cfg`.

## Against a DGX Spark

The same prompts, seed, size, steps, CFG and sampler on a DGX Spark (GB10), warm, but Euler in the stable-diffusion.cpp row:

| DGX Spark setup | Prompt repeated | New prompt | luce_server is |
|---|---:|---:|---:|
| ComfyUI, int8 convrot diffusion weights from our converter, 16-bit attention, full VAE | 4.54 s | 5.60 s | **1.5× / 1.8×** |
| ComfyUI, its shipped files: int8 attention, small VAE | 4.22 s | 5.34 s | 1.4× / 1.7× |
| stable-diffusion.cpp `b13a482` (CUDA), same int8 weights, Euler | | 16.1 s | 5.3× |

- ComfyUI 0.42.0 (`b1a247c`), PyTorch 2.14.1 with CUDA 13.0, comfy-kitchen 0.2.39, its native MOSS-Image nodes and its default sampler for the model, er_sde on the simple schedule. Its text encoder is the shipped Qwen3.5 9B w4a8 in both ComfyUI rows; ours is Q8_0.
- Times are ComfyUI's own "Prompt executed in", which includes saving the PNG but not the HTTP round trip. ComfyUI keeps the previous request's text encoding, so the same prompt twice in a row skips it: one such request per setup, against five new-prompt requests (5.59-5.61 s).
- The R9700 side is luce_server's request time with a new prompt each time (3.06 s), so the comparison counts more work on our side, not less.
- ComfyUI per new-prompt image: text encoding 1.07 s, sampling 3.95 s (0.49 s a step), VAE 0.45 s, saving 0.12-0.29 s. luce_server: 0.03 s, 2.82 s (0.35 s a step), 0.18 s, and 0.03 s for the PNG and HTTP.
- The shipped files trade quality for speed: in our poster test they garbled the word "LUCEBOX", which the 16-bit attention run kept readable (one prompt, one seed).

## Power

The R9700 runs at its power limit during every request. These numbers use 330 W, the most the driver allows; the card ships at 300 W, which makes an image about 1% slower.
