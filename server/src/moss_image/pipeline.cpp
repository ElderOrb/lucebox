#include "pipeline.h"

#include "blas.h"
#include "kernels.h"
#include "text_encoder.h"
#include "weights.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <iterator>
#include <list>
#include <string>
#include <vector>

namespace luce::moss_image {

namespace {

constexpr float   kEps         = 1e-6f;
constexpr size_t  kTextCache   = 4;    // prompts whose text rows are kept (stable-diffusion.cpp keeps 4 too)
constexpr int     kSkipLayer   = 22;   // the unconditional pass skips it, as stable-diffusion.cpp runs MOSS-Image
constexpr int     kGroups      = 32;   // VAE GroupNorm groups
constexpr int     kMaxStepRows = 16;   // linear_bf16_rows takes up to 16 steps at once

int64_t round_up(const int64_t v, const int64_t m) {
    return (v + m - 1) / m * m;
}

// A device buffer that only grows: work buffers are sized for the largest request so far. Grown only while the
// stream is idle.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { (void) hipFree(ptr_); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer & operator=(const DeviceBuffer &) = delete;

    bool reserve(const size_t bytes) {
        if (bytes <= bytes_) {
            return true;
        }
        (void) hipFree(ptr_);
        ptr_   = nullptr;
        bytes_ = 0;
        if (hipMalloc(&ptr_, bytes) != hipSuccess) {
            ptr_ = nullptr;
            return false;
        }
        bytes_ = bytes;
        return true;
    }

    template <typename T> T * as() const { return static_cast<T *>(ptr_); }

private:
    void * ptr_   = nullptr;
    size_t bytes_ = 0;
};

// Pinned host memory for the per-request uploads (RoPE tables, timestep features).
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    ~PinnedBuffer() { (void) hipHostFree(ptr_); }
    PinnedBuffer(const PinnedBuffer &) = delete;
    PinnedBuffer & operator=(const PinnedBuffer &) = delete;

    float * reserve(const size_t count) {
        if (count > count_) {
            (void) hipHostFree(ptr_);
            ptr_   = nullptr;
            count_ = 0;
            if (hipHostMalloc(reinterpret_cast<void **>(&ptr_), count * sizeof(float)) != hipSuccess) {
                ptr_ = nullptr;
                return nullptr;
            }
            count_ = count;
        }
        return ptr_;
    }

private:
    float * ptr_   = nullptr;
    size_t  count_ = 0;
};

// A prompt's text rows of the DiT input: the encoder's hidden states through the text embedder.
struct TextRows {
    std::string  prompt;
    bool         valid = false;
    int64_t      n     = 0;
    DeviceBuffer rows;  // [n][kHidden] f32
};

} // namespace

struct Pipeline::Impl {
    hipStream_t stream = nullptr;
    Weights     weights;
    TextEncoder encoder;
    Blas        blas;

    // The text rows of the last kTextCache prompts and negative prompts, the most recent first: a request that
    // repeats one (another seed or size) skips the text encoder.
    std::list<TextRows> text_cache;
    const TextRows *    cond   = nullptr;  // this request's prompt rows, in text_cache
    const TextRows *    uncond = nullptr;  // and its negative prompt's, with CFG

    // Text embedder work buffers.
    DeviceBuffer t_bf16, t_mm, t_q, t_k, t_v, t_att, t_pe;
    // Timestep path, every step of a request at once.
    DeviceBuffer time_in, time_hidden, time_emb, block_mod, head_mod;
    // DiT.
    DeviceBuffer x, xq, xs, q, k, v, att, sums, hq, hs, pe, state_bf16, img, head_in, pred, state, old_den, old_dd;
    // VAE.
    DeviceBuffer maps[4], nhwc[2], gn_scratch, scores, probs, rgb;
    PinnedBuffer host_pe, host_time;
    hipEvent_t   events[4] = {};
    bool         warming_up = false;  // the warm-up images log nothing
    std::string  tuning_path;         // where the tuned GEMM kernels are kept; empty: nowhere

    ~Impl() {
        for (hipEvent_t e : events) {
            if (e != nullptr) {
                (void) hipEventDestroy(e);
            }
        }
        if (stream != nullptr) {
            (void) hipStreamDestroy(stream);
        }
    }

    bool text(const std::string & prompt, const TextRows *& found, std::string & error);
    bool reserve(int64_t n_img, int64_t rows, int n_seq, int steps, int64_t latent_hw, std::string & error);
    bool dit_layer(const DitLayer & layer, const float * mod, int64_t active, int64_t rows_c, int64_t rows_u);
    bool sample(const ImageRequest & request, int lh, int lw, int64_t rows_c, int64_t rows_u, std::string & error);
    void resnet(const VaeResBlock & block, const float * in, int ic, int oc, int w, int h, float * tmp, float * tmp2,
                float * out);
    bool attention(const VaeAttention & block, const float * in, int w, int h, float * out);
    bool decode(int lh, int lw, std::string & error);
    void save_tuning();
};

// ---------------------------------------------------------------------------------------------------------------
// Text

bool Pipeline::Impl::text(const std::string & prompt, const TextRows *& found, std::string & error) {
    for (auto it = text_cache.begin(); it != text_cache.end(); ++it) {
        if (it->valid && it->prompt == prompt) {
            text_cache.splice(text_cache.begin(), text_cache, it);
            found = &text_cache.front();
            return true;
        }
    }
    // A new prompt takes the least recently used entry, never the one the request's other prompt just found.
    if (text_cache.size() < kTextCache) {
        text_cache.emplace_front();
    } else {
        text_cache.splice(text_cache.begin(), text_cache, std::prev(text_cache.end()));
    }
    TextRows & out = text_cache.front();
    out.valid      = false;
    TextHidden hidden;
    if (!encoder.encode(encoder.tokenize(prompt), hidden, error)) {
        return false;
    }
    const int64_t n = hidden.n_tokens;
    if (!t_bf16.reserve(n * kFfn * sizeof(uint16_t)) || !t_mm.reserve(n * 2 * kFfn * sizeof(float)) ||
        !t_q.reserve(n * kHidden * sizeof(float)) || !t_k.reserve(n * kHidden * sizeof(uint16_t)) ||
        !t_v.reserve(n * kHidden * sizeof(uint16_t)) || !t_att.reserve(n * kHidden * sizeof(float)) ||
        !t_pe.reserve(n * kRopePairs * 2 * sizeof(float)) || !out.rows.reserve(n * kHidden * sizeof(float))) {
        error = "out of GPU memory for the text embedder";
        return false;
    }
    float * pe = host_pe.reserve(size_t(n) * kRopePairs * 2);
    if (pe == nullptr) {
        error = "out of pinned host memory";
        return false;
    }
    rope_table(0, 0, int(n), pe);  // the text rows: positions (0, i, 0, 0)
    (void) hipMemcpyAsync(t_pe.as<float>(), pe, n * kRopePairs * 2 * sizeof(float), hipMemcpyHostToDevice, stream);

    const DitWeights & w  = weights.dit;
    uint16_t *         b  = t_bf16.as<uint16_t>();
    float *            mm = t_mm.as<float>();
    float *            tx = out.rows.as<float>();
    bool               ok = true;
    f32_to_bf16(hidden.data, n * kTextHidden, b, stream);
    ok = ok && blas.gemm(GemmType::bf16_f32, w.text_proj.w, kTextHidden, b, kTextHidden, tx, kHidden, kHidden, n,
                         kTextHidden);
    add_bias(tx, w.text_proj.b, n, kHidden, tx, stream);
    for (const TextBlock & block : w.text_blocks) {
        rms_norm_mul_bf16(tx, block.norm1, kEps, n, kHidden, b, stream);
        ok = ok && blas.gemm(GemmType::bf16_f32, block.qkv.w, kHidden, b, kHidden, mm, 3 * kHidden, 3 * kHidden, n,
                             kHidden);
        text_qkv_prologue(mm, block.qkv.b, block.q_norm, block.k_norm, kEps, t_pe.as<float>(), n, kHeads,
                          t_q.as<float>(), t_k.as<uint16_t>(), t_v.as<uint16_t>(), stream);
        attention_d128(t_q.as<float>(), t_k.as<uint16_t>(), t_v.as<uint16_t>(), n, 0, n, kHeads,
                       1.0f / std::sqrt(float(kHeadDim)), t_att.as<float>(), stream);
        f32_to_bf16(t_att.as<float>(), n * kHidden, b, stream);
        ok = ok && blas.gemm(GemmType::bf16_f32, block.proj.w, kHidden, b, kHidden, mm, kHidden, kHidden, n, kHidden);
        add_bias_residual(mm, block.proj.b, n, kHidden, tx, stream);
        rms_norm_mul_bf16(tx, block.norm2, kEps, n, kHidden, b, stream);
        ok = ok && blas.gemm(GemmType::bf16_f32, block.up_gate.w, kHidden, b, kHidden, mm, 2 * kFfn, 2 * kFfn, n,
                             kHidden);
        text_situ_bf16(mm, block.up_gate.b, block.up_gate.b + kFfn, n, kFfn, b, stream);
        ok = ok && blas.gemm(GemmType::bf16_f32, block.down.w, kFfn, b, kFfn, mm, kHidden, kHidden, n, kFfn);
        add_bias_residual(mm, block.down.b, n, kHidden, tx, stream);
    }
    // The encoder reuses its output buffer on the next prompt: let this one's readers finish first.
    const hipError_t status = hipStreamSynchronize(stream);
    if (!ok || status != hipSuccess) {
        error = ok ? std::string("text embedder: ") + hipGetErrorString(status) : "text embedder: a GEMM failed";
        return false;
    }
    out.prompt = prompt;
    out.n      = n;
    out.valid  = true;
    found      = &out;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Buffers

bool Pipeline::Impl::reserve(const int64_t n_img, const int64_t rows, const int n_seq, const int steps,
                             const int64_t latent_hw, std::string & error) {
    const int64_t rows_pad = round_up(rows, 4);  // the hipBLASLt int8 operand's rows
    bool ok = time_in.reserve(size_t(steps) * kTimeFreqs * sizeof(float)) &&
              time_hidden.reserve(size_t(steps) * kHidden * sizeof(float)) &&
              time_emb.reserve(size_t(steps) * kHidden * sizeof(float)) &&
              block_mod.reserve(size_t(steps) * 6 * kHidden * sizeof(float)) &&
              head_mod.reserve(size_t(steps) * 2 * kHidden * sizeof(float));
    ok = ok && x.reserve(rows * kHidden * sizeof(float)) && xq.reserve(rows_pad * kHidden) &&
         xs.reserve(rows_pad * sizeof(float)) && q.reserve(rows * kHidden * sizeof(float)) &&
         k.reserve(rows * kHidden * sizeof(uint16_t)) && v.reserve(rows * kHidden * sizeof(uint16_t)) &&
         att.reserve(rows * kHidden * sizeof(float)) && sums.reserve(rows_pad * 2 * kFfn * sizeof(int32_t)) &&
         hq.reserve(rows_pad * kFfn) && hs.reserve(rows_pad * sizeof(float)) &&
         pe.reserve(rows * kRopePairs * 2 * sizeof(float)) &&
         state_bf16.reserve(n_img * kLatentChannels * sizeof(uint16_t)) &&
         img.reserve(n_img * kHidden * sizeof(float)) && head_in.reserve(n_seq * n_img * kHidden * sizeof(uint16_t)) &&
         pred.reserve(n_seq * n_img * kLatentChannels * sizeof(float)) &&
         state.reserve(n_img * kLatentChannels * sizeof(float)) &&
         old_den.reserve(n_img * kLatentChannels * sizeof(float)) &&
         old_dd.reserve(n_img * kLatentChannels * sizeof(float));
    // VAE decoder: the largest map is 512 channels at 4x4 or 256 channels at 8x8 the latent's pixels, so 16384
    // values per latent pixel; the mid-block attention scores are latent_hw^2.
    size_t gn = 0;
    for (const auto & [c, scale] : { std::pair{ 512, 1 }, { 512, 4 }, { 512, 16 }, { 256, 16 }, { 256, 64 },
                                   { 128, 64 } }) {
        gn = std::max(gn, group_norm_scratch_bytes(c, latent_hw * scale, kGroups));
    }
    for (DeviceBuffer & map : maps) {
        ok = ok && map.reserve(latent_hw * 16384 * sizeof(float));
    }
    for (DeviceBuffer & map : nhwc) {
        ok = ok && map.reserve(latent_hw * 16384 * sizeof(uint16_t));
    }
    ok = ok && gn_scratch.reserve(gn) && scores.reserve(latent_hw * latent_hw * sizeof(float)) &&
         probs.reserve(latent_hw * latent_hw * sizeof(uint16_t)) && rgb.reserve(latent_hw * 64 * 3);
    if (!ok) {
        error = "out of GPU memory for this image size";
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// DiT

// One DiT block over the first `active` rows of x: the conditional sequence (rows_c rows), then the unconditional
// one (rows_u rows) unless this block skips it.
bool Pipeline::Impl::dit_layer(const DitLayer & layer, const float * mod, const int64_t active, const int64_t rows_c,
                               const int64_t rows_u) {
    const float * shift1 = mod;
    const float * scale1 = mod + kHidden;
    const float * gate1  = mod + 2 * kHidden;
    const float * shift2 = mod + 3 * kHidden;
    const float * scale2 = mod + 4 * kHidden;
    const float * gate2  = mod + 5 * kHidden;
    float *       xf     = x.as<float>();

    norm_quantize_i8(xf, active, kHidden, layer.norm1, scale1, shift1, kEps, xq.as<int8_t>(), xs.as<float>(), stream);
    gemm_i8_qkv_attention(layer.qkv.w, layer.qkv.scale, xq.as<int8_t>(), xs.as<float>(), active, kHidden, kHeads,
                          layer.q_norm, layer.k_norm, kEps, pe.as<float>(), q.as<float>(), k.as<uint16_t>(),
                          v.as<uint16_t>(), stream);
    const float attn_scale = 1.0f / std::sqrt(float(kHeadDim));
    attention_d128(q.as<float>(), k.as<uint16_t>(), v.as<uint16_t>(), active, 0, rows_c, kHeads, attn_scale,
                   att.as<float>(), stream);
    if (active > rows_c) {
        attention_d128(q.as<float>(), k.as<uint16_t>(), v.as<uint16_t>(), active, rows_c, rows_u, kHeads, attn_scale,
                       att.as<float>(), stream);
    }
    quantize_i8(att.as<float>(), active, kHidden, xq.as<int8_t>(), xs.as<float>(), stream);
    gemm_i8_gated_residual(layer.o.w, layer.o.scale, xq.as<int8_t>(), xs.as<float>(), active, kHidden, kHidden, gate1,
                           xf, stream);
    norm_quantize_i8(xf, active, kHidden, layer.norm2, scale2, shift2, kEps, xq.as<int8_t>(), xs.as<float>(), stream);
    const bool ok = blas.gemm(GemmType::i8_i32, layer.up_gate.w, kHidden, xq.as<int8_t>(), kHidden,
                              sums.as<int32_t>(), 2 * kFfn, 2 * kFfn, round_up(active, 4), kHidden);
    situ_quantize_i8(sums.as<int32_t>(), xs.as<float>(), layer.up_gate.scale, active, kFfn, hq.as<int8_t>(),
                     hs.as<float>(), stream);
    gemm_i8_gated_residual(layer.down.w, layer.down.scale, hq.as<int8_t>(), hs.as<float>(), active, kFfn, kHidden,
                           gate2, xf, stream);
    return ok;
}

// The sampling loop: the latent state after the last step, in `state`.
bool Pipeline::Impl::sample(const ImageRequest & request, const int lh, const int lw, const int64_t rows_c,
                            const int64_t rows_u, std::string & error) {
    const DitWeights & w      = weights.dit;
    const int64_t      n_img  = int64_t(lh) * lw;
    const int64_t      rows   = rows_c + rows_u;
    const int          n_seq  = rows_u > 0 ? 2 : 1;
    const uint64_t     seed   = uint64_t(request.seed);
    const Schedule     sched  = request.scheduler == Scheduler::simple ? Schedule::Simple : Schedule::Discrete;
    const Sampler      method = request.sample_method == SampleMethod::euler ? Sampler::Euler : Sampler::ErSde;

    const std::vector<float>       sigmas = flow_sigmas(sched, request.steps, request.flow_shift);
    const std::vector<SamplerStep> plan   = plan_steps(method, sigmas, request.cfg_scale, request.eta);
    const int                      steps  = int(plan.size());  // Euler skips a step whose sigma does not change

    // The timestep path of every step at once: the modulation of step s is row s.
    float * features = host_time.reserve(size_t(steps) * kTimeFreqs);
    float * table    = host_pe.reserve(size_t(rows) * kRopePairs * 2);
    if (features == nullptr || table == nullptr) {
        error = "out of pinned host memory";
        return false;
    }
    for (int s = 0; s < steps; ++s) {
        time_features(plan[s].sigma, features + size_t(s) * kTimeFreqs);
    }
    (void) hipMemcpyAsync(time_in.as<float>(), features, size_t(steps) * kTimeFreqs * sizeof(float),
                          hipMemcpyHostToDevice, stream);
    for (int s0 = 0; s0 < steps; s0 += kMaxStepRows) {
        const int n = std::min(kMaxStepRows, steps - s0);
        linear_bf16_rows(w.time_in.w, w.time_in.b, time_in.as<float>() + size_t(s0) * kTimeFreqs, n, kHidden,
                         kTimeFreqs, true, time_hidden.as<float>() + size_t(s0) * kHidden, stream);
        linear_bf16_rows(w.time_out.w, w.time_out.b, time_hidden.as<float>() + size_t(s0) * kHidden, n, kHidden,
                         kHidden, false, time_emb.as<float>() + size_t(s0) * kHidden, stream);
        linear_bf16_rows(w.block_modulation.w, w.block_modulation.b, time_emb.as<float>() + size_t(s0) * kHidden, n,
                         6 * kHidden, kHidden, false, block_mod.as<float>() + size_t(s0) * 6 * kHidden, stream);
        linear_bf16_rows(w.head_modulation.w, w.head_modulation.b, time_emb.as<float>() + size_t(s0) * kHidden, n,
                         2 * kHidden, kHidden, false, head_mod.as<float>() + size_t(s0) * 2 * kHidden, stream);
    }

    // RoPE tables of both sequences, stacked like their rows.
    rope_table(lh, lw, int(rows_c - n_img), table);
    if (rows_u > 0) {
        rope_table(lh, lw, int(rows_u - n_img), table + rows_c * kRopePairs * 2);
    }
    (void) hipMemcpyAsync(pe.as<float>(), table, rows * kRopePairs * 2 * sizeof(float), hipMemcpyHostToDevice,
                          stream);

    init_noise(seed, n_img, kLatentChannels, state.as<float>(), stream);
    bool ok = true;
    for (int s = 0; s < steps; ++s) {
        f32_to_bf16(state.as<float>(), n_img * kLatentChannels, state_bf16.as<uint16_t>(), stream);
        ok = ok && blas.gemm(GemmType::bf16_f32, w.video_embedder.w, kLatentChannels, state_bf16.as<uint16_t>(),
                             kLatentChannels, img.as<float>(), kHidden, kHidden, n_img, kLatentChannels);
        assemble_tokens(img.as<float>(), w.video_embedder.b, n_img, cond->rows.as<float>(), cond->n,
                        rows_u > 0 ? uncond->rows.as<float>() : nullptr, rows_u > 0 ? uncond->n : 0, kHidden,
                        x.as<float>(), stream);
        const float * mod = block_mod.as<float>() + size_t(s) * 6 * kHidden;
        for (int l = 0; l < kLayers; ++l) {
            ok = ok && dit_layer(w.layers[l], mod, l == kSkipLayer ? rows_c : rows, rows_c, rows_u);
        }
        final_head_bf16(x.as<float>(), kHidden, n_img, n_seq, rows_c, w.final_norm,
                        head_mod.as<float>() + size_t(s) * 2 * kHidden, kEps, head_in.as<uint16_t>(), stream);
        ok = ok && blas.gemm(GemmType::bf16_f32, w.final_linear.w, kHidden, head_in.as<uint16_t>(), kHidden,
                             pred.as<float>(), kLatentChannels, kLatentChannels, n_seq * n_img, kHidden);
        sampler_step(plan[s], pred.as<float>(), n_seq == 2 ? pred.as<float>() + n_img * kLatentChannels : nullptr,
                     seed, n_img, kLatentChannels, state.as<float>(), old_den.as<float>(), old_dd.as<float>(), stream);
    }
    if (!ok) {
        error = "DiT: a GEMM failed";
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// VAE decoder

// GroupNorm and SiLU, 3x3 convolution, twice, plus the shortcut (a 1x1 convolution where the channels change).
// in [ic][hw] -> out [oc][hw]; tmp and tmp2 are scratch maps.
void Pipeline::Impl::resnet(const VaeResBlock & block, const float * in, const int ic, const int oc, const int w,
                            const int h, float * tmp, float * tmp2, float * out) {
    const int64_t hw  = int64_t(w) * h;
    uint16_t *    n16 = nhwc[0].as<uint16_t>();
    group_norm_nhwc(in, ic, hw, kGroups, kEps, block.norm1.w, block.norm1.b, true, gn_scratch.as<void>(), n16, stream);
    conv2d(n16, w, h, ic, false, block.conv1.w, 3, 3, oc, 1, 1, block.conv1.b, nullptr, tmp, stream);
    group_norm_nhwc(tmp, oc, hw, kGroups, kEps, block.norm2.w, block.norm2.b, true, gn_scratch.as<void>(), n16, stream);
    if (block.nin.w == nullptr) {
        conv2d(n16, w, h, oc, false, block.conv2.w, 3, 3, oc, 1, 1, block.conv2.b, in, out, stream);
        return;
    }
    conv2d(n16, w, h, oc, false, block.conv2.w, 3, 3, oc, 1, 1, block.conv2.b, nullptr, tmp2, stream);
    uint16_t * in16 = nhwc[1].as<uint16_t>();
    to_nhwc_f16(in, ic, hw, in16, stream);
    conv2d(in16, w, h, ic, false, block.nin.w, 1, 1, oc, 0, 0, block.nin.b, tmp2, out, stream);
}

// The mid-block self-attention, one 512-wide head over every pixel: in [512][hw] -> out [512][hw].
bool Pipeline::Impl::attention(const VaeAttention & block, const float * in, const int w, const int h, float * out) {
    constexpr int C   = 512;
    const int64_t hw  = int64_t(w) * h;
    uint16_t *    n16 = nhwc[0].as<uint16_t>();
    float *       qkv = maps[2].as<float>();  // [3 * C][hw]
    group_norm_nhwc(in, C, hw, kGroups, kEps, block.norm.w, block.norm.b, false, gn_scratch.as<void>(), n16, stream);
    conv2d(n16, w, h, C, false, block.qkv.w, 1, 1, 3 * C, 0, 0, block.qkv.b, nullptr, qkv, stream);
    uint16_t * qk = nhwc[1].as<uint16_t>();  // [hw][2 * C]: q, then k
    uint16_t * vt = n16;                     // [C][hw]
    to_nhwc_f16(qkv, 2 * C, hw, qk, stream);
    f32_to_f16(qkv + 2 * C * hw, C * hw, vt, stream);
    bool ok = blas.gemm(GemmType::f16_f32, qk + C, 2 * C, qk, 2 * C, scores.as<float>(), hw, hw, hw, C);
    softmax_f16(scores.as<float>(), hw, hw, 1.0f / std::sqrt(float(C)), probs.as<uint16_t>(), stream);
    float * o = maps[3].as<float>();  // [hw][C]
    ok = ok && blas.gemm(GemmType::f16_f32, vt, hw, probs.as<uint16_t>(), hw, o, C, C, hw, hw);
    f32_to_f16(o, hw * C, qk, stream);
    conv2d(qk, w, h, C, false, block.proj.w, 1, 1, C, 0, 0, block.proj.b, in, out, stream);
    return ok;
}

// The latent state through the VAE decoder into rgb [16 lh][16 lw][3] uint8.
bool Pipeline::Impl::decode(const int lh, const int lw, std::string & error) {
    const VaeDecoder & d = weights.vae;
    int                h = 2 * lh, w = 2 * lw;  // latent pixels
    int64_t            hw = int64_t(h) * w;
    float *            a  = maps[0].as<float>();
    float *            b  = maps[1].as<float>();
    float *            t1 = maps[2].as<float>();
    float *            t2 = maps[3].as<float>();
    uint16_t *         n16 = nhwc[0].as<uint16_t>();

    unpack_latent(state.as<float>(), d.latents_mean, d.latents_std, lh, lw, a, stream);
    to_nhwc_f16(a, 32, hw, n16, stream);
    conv2d(n16, w, h, 32, false, d.post_quant.w, 1, 1, 32, 0, 0, d.post_quant.b, nullptr, b, stream);
    to_nhwc_f16(b, 32, hw, n16, stream);
    conv2d(n16, w, h, 32, false, d.conv_in.w, 3, 3, 512, 1, 1, d.conv_in.b, nullptr, a, stream);

    resnet(d.mid_block1, a, 512, 512, w, h, t1, t2, b);
    bool ok = attention(d.mid_attn, b, w, h, a);
    resnet(d.mid_block2, a, 512, 512, w, h, t1, t2, b);
    int channels = 512;
    for (int level = 3; level >= 0; --level) {
        const int out_channels = level >= 2 ? 512 : level == 1 ? 256 : 128;
        for (const VaeResBlock & block : d.up_blocks[level]) {
            resnet(block, b, channels, out_channels, w, h, t1, t2, a);
            std::swap(a, b);
            channels = out_channels;
        }
        if (d.upsample[level].w != nullptr) {
            to_nhwc_f16(b, channels, hw, n16, stream);
            h *= 2;
            w *= 2;
            hw *= 4;
            conv2d(n16, w, h, channels, true, d.upsample[level].w, 3, 3, channels, 1, 1, d.upsample[level].b, nullptr,
                   a, stream);
            std::swap(a, b);
        }
    }
    group_norm_nhwc(b, channels, hw, kGroups, kEps, d.norm_out.w, d.norm_out.b, true, gn_scratch.as<void>(), n16,
                    stream);
    conv2d(n16, w, h, channels, false, d.conv_out.w, 3, 3, 3, 1, 1, d.conv_out.b, nullptr, a, stream);
    to_rgb8(a, hw, rgb.as<uint8_t>(), stream);
    if (!ok) {
        error = "VAE: a GEMM failed";
    }
    return ok;
}

// Kernels tuned since the tuning file was read or written go into it. A file that cannot be written is said once.
void Pipeline::Impl::save_tuning() {
    if (!tuning_path.empty() && blas.tuning_changed() && !blas.save_tuning(tuning_path)) {
        std::fprintf(stderr, "[moss-image] cannot write the GEMM tuning file %s\n", tuning_path.c_str());
        tuning_path.clear();
    }
}

// ---------------------------------------------------------------------------------------------------------------

Pipeline::Pipeline() : impl_(std::make_unique<Impl>()) {}

Pipeline::~Pipeline() = default;

bool Pipeline::load(const PipelineOptions & options, std::string & error) {
    Impl & m      = *impl_;
    int    device = 0;
    if (hipGetDevice(&device) != hipSuccess || !device_supported(device)) {
        error = "MOSS-Image needs an RDNA4 GPU (gfx1200 or gfx1201)";
        return false;
    }
    if (hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking) != hipSuccess || !m.blas.init(m.stream)) {
        error = "cannot create the HIP stream or the hipBLASLt handle";
        return false;
    }
    for (hipEvent_t & e : m.events) {
        if (hipEventCreate(&e) != hipSuccess) {
            error = "cannot create HIP events";
            return false;
        }
    }
    using clock        = std::chrono::steady_clock;
    const auto seconds = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    // GEMM kernels tuned by an earlier run need no warm-up image: they are found again, and their first products run,
    // beside the uploads on a thread and stream of their own.
    m.tuning_path      = options.tuning_path;
    const size_t tuned = m.tuning_path.empty() ? 0 : m.blas.load_tuning(m.tuning_path);
    // The two files load at once: the text encoder through ggml on its own thread, the model through our uploads.
    const auto  t0 = clock::now();
    std::string encoder_error;
    double      encoder_seconds = 0.0;
    auto        encoder         = std::async(std::launch::async, [&] {
        const bool ok   = m.encoder.load(options.text_encoder_path, encoder_error);
        encoder_seconds = seconds(t0, clock::now());
        return ok;
    });
    auto        kernels         = std::async(std::launch::async, [&] {
        hipStream_t stream = nullptr;
        if (tuned == 0 || hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) {
            return size_t(0);
        }
        const size_t ran = m.blas.preload(stream);
        (void) hipStreamDestroy(stream);
        return ran;
    });
    const bool   model_ok      = m.weights.load(options.model_path, m.stream, error);
    const double model_seconds = seconds(t0, clock::now());
    const bool   encoder_ok    = encoder.get();
    const size_t preloaded     = kernels.get();
    if (!model_ok) {
        return false;
    }
    if (!encoder_ok) {
        error = encoder_error;
        return false;
    }
    const auto t1 = clock::now();
    if (tuned > 0 && preloaded == tuned) {
        // The encoder's first prompt loads its kernels, which the first request then finds loaded.
        TextHidden hidden;
        if (!m.encoder.encode(m.encoder.tokenize("a photo of a lighthouse"), hidden, error)) {
            return false;
        }
        std::fprintf(stderr,
                     "[moss-image] loaded in %.1f s: model %.1f s and text encoder %.1f s at once, with the %zu GEMM "
                     "kernels tuned in %s, first prompt %.2f s\n",
                     seconds(t0, clock::now()), model_seconds, encoder_seconds, preloaded, m.tuning_path.c_str(),
                     seconds(t1, clock::now()));
        return true;
    }
    // One image with and one without CFG at the warm-up size: every GEMM shape of a request is tuned before the first
    // (only those not found again, after a partial preload), and the tuning file keeps the kernels for the next run.
    ImageRequest warmup;
    warmup.prompt = "a photo of a lighthouse";
    warmup.width  = options.warmup_width;
    warmup.height = options.warmup_height;
    warmup.steps  = 1;
    warmup.seed   = 0;
    RgbImage image;
    m.warming_up = true;
    for (const float cfg : { 2.5f, 1.0f }) {
        warmup.cfg_scale = cfg;
        if (!generate(warmup, image, error)) {
            error = "warm-up: " + error;
            return false;
        }
    }
    m.warming_up = false;
    std::fprintf(stderr,
                 "[moss-image] loaded in %.1f s: model %.1f s and text encoder %.1f s at once, warm-up %.1f s\n",
                 seconds(t0, clock::now()), model_seconds, encoder_seconds, seconds(t1, clock::now()));
    m.save_tuning();
    return true;
}

bool Pipeline::generate(const ImageRequest & request, RgbImage & image, std::string & error) {
    Impl & m = *impl_;
    if (request.width % 16 != 0 || request.height % 16 != 0 ||
        std::min(request.width, request.height) < kMinImageSide ||
        int64_t(request.width) * request.height > kMaxImagePixels) {
        error = "width and height must be multiples of 16, at least " + std::to_string(kMinImageSide) +
                ", and at most " + std::to_string(kMaxImagePixels) + " pixels together";
        return false;
    }
    if (request.steps < 1 || request.steps > kMaxImageSteps) {
        error = "steps must be 1 to " + std::to_string(kMaxImageSteps);
        return false;
    }
    if (!(request.cfg_scale >= 0.0f) || !std::isfinite(request.cfg_scale) || !(request.flow_shift > 0.0f) ||
        !std::isfinite(request.flow_shift) || !(request.eta >= 0.0f) || !std::isfinite(request.eta) ||
        request.seed < 0) {
        error = "flow_shift must be positive; cfg_scale, eta and the seed must not be negative";
        return false;
    }
    const bool    cfg   = request.cfg_scale != 1.0f;
    const int     lh    = request.height / 16;
    const int     lw    = request.width / 16;
    const int64_t n_img = int64_t(lh) * lw;

    (void) hipEventRecord(m.events[0], m.stream);
    if (!m.text(request.prompt, m.cond, error) || (cfg && !m.text(request.negative_prompt, m.uncond, error))) {
        return false;
    }
    const int64_t rows_c = n_img + m.cond->n;
    const int64_t rows_u = cfg ? n_img + m.uncond->n : 0;
    if (!m.reserve(n_img, rows_c + rows_u, cfg ? 2 : 1, request.steps, 4 * n_img, error)) {
        return false;
    }
    (void) hipEventRecord(m.events[1], m.stream);
    if (!m.sample(request, lh, lw, rows_c, rows_u, error)) {
        return false;
    }
    (void) hipEventRecord(m.events[2], m.stream);
    if (!m.decode(lh, lw, error)) {
        return false;
    }
    image.width  = request.width;
    image.height = request.height;
    image.rgb.resize(size_t(request.width) * request.height * 3);
    (void) hipMemcpyAsync(image.rgb.data(), m.rgb.as<uint8_t>(), image.rgb.size(), hipMemcpyDeviceToHost, m.stream);
    (void) hipEventRecord(m.events[3], m.stream);
    hipError_t status = hipStreamSynchronize(m.stream);
    if (status == hipSuccess) {
        status = hipGetLastError();
    }
    if (status != hipSuccess) {
        error = std::string("GPU error: ") + hipGetErrorString(status);
        return false;
    }
    if (m.warming_up) {
        return true;
    }
    float text_ms = 0.0f, sample_ms = 0.0f, decode_ms = 0.0f;
    (void) hipEventElapsedTime(&text_ms, m.events[0], m.events[1]);
    (void) hipEventElapsedTime(&sample_ms, m.events[1], m.events[2]);
    (void) hipEventElapsedTime(&decode_ms, m.events[2], m.events[3]);
    std::fprintf(stderr, "[moss-image] %dx%d, %d steps, cfg %.2f: text %.0f ms, sampling %.0f ms, decode %.0f ms\n",
                 request.width, request.height, request.steps, request.cfg_scale, text_ms, sample_ms, decode_ms);
    m.save_tuning();
    return true;
}

} // namespace luce::moss_image
