// MOSS-Image text to image on one RDNA4 GPU: the Qwen3.5-9B text encoder, the int8 DiT sampling loop and the FLUX.2
// VAE decoder, all on one stream.
//
// A request with CFG runs the conditional and the unconditional sequence as one batch of rows through every GEMM;
// only attention and the rotary tables stay per sequence. The text side of a prompt (encoder, projection and the two
// text blocks) does not depend on the step, so it runs once per prompt and the last prompt and the negative prompt
// stay cached. Buffers are sized for the largest request so far and reused.

#pragma once

#include "image_server.h"

#include <memory>
#include <string>

namespace luce::moss_image {

struct PipelineOptions {
    std::string model_path;         // the moss-image GGUF: DiT and VAE (scripts/convert_moss_image.py)
    std::string text_encoder_path;  // the Qwen3.5-9B GGUF
    int         warmup_width  = 1024;
    int         warmup_height = 1024;
};

class Pipeline final : public Generator {
public:
    Pipeline();
    ~Pipeline() override;
    Pipeline(const Pipeline &) = delete;
    Pipeline & operator=(const Pipeline &) = delete;

    // Weights on the GPU, then one warm-up image at the warm-up size so the first request pays no GEMM tuning;
    // false with `error` set.
    bool load(const PipelineOptions & options, std::string & error);

    bool generate(const ImageRequest & request, RgbImage & image, std::string & error) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace luce::moss_image
