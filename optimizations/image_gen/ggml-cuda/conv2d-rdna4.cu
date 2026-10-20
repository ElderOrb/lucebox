#include "conv2d-rdna4.cuh"
#include "group-norm-fused.cuh"

// 2D convolution (stride 1, dilation 1) as an implicit GEMM on RDNA4 WMMA:
// out[oc][p] = sum over taps t and input channels ci of w[oc][t][ci] * x[p + t][ci], f16 operands, f32 sums.
// The input is first written as f16 channels-last and the weights as [OC][KH * KW][IC], so both WMMA operands are rows
// of 8 consecutive channels (one 16-byte load per lane). A block computes 128 output channels x 128 output pixels with
// 8 waves of 32 x 64; the K loop takes 32 channels of one tap per step, double-buffered in LDS with one barrier per
// step. Pixels outside the input read as zero (the padding). The epilogue adds the bias, then the residual, and writes
// f32 NCHW. With UP the input is a nearest 2x upscale read from its half-size source.

namespace {

typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float    float8_t __attribute__((ext_vector_type(8)));

constexpr int BM      = 128;     // output channels per block
constexpr int BN      = 128;     // output pixels per block
constexpr int BK      = 32;      // input channels per K step
constexpr int THREADS = 256;
constexpr int STRIDE  = BK + 8;  // halves per LDS row, padded against bank conflicts
constexpr int SM_BM   = 16;      // output channels of the small-OC kernel (one WMMA tile)

struct conv2d_rdna4_params {
    const _Float16 * x;     // [N][IH * IW][IC], with UP [N][IH / 2 * IW / 2][IC]
    const _Float16 * w;     // [OC][KH * KW][IC]
    const float *    bias;  // [OC] or null
    const float *    res;   // [N][OC][OH * OW] or null
    float *          y;     // [N][OC][OH * OW]
    int              IW, IH, OW, OH, KW, KH, PX, PY, IC, OC;
};

template <bool UP>
__device__ __forceinline__ uint4 load_pixel(const conv2d_rdna4_params & p, const _Float16 * xn, const bool in, const int iy,
                                            const int ix, const int ci) {
    if (!in) {
        return make_uint4(0, 0, 0, 0);
    }
    const int64_t pix = UP ? (int64_t) (iy / 2) * (p.IW / 2) + ix / 2 : (int64_t) iy * p.IW + ix;
    return *(const uint4 *) (xn + pix * p.IC + ci);
}

template <bool UP>
__global__ void __launch_bounds__(THREADS) conv2d_rdna4_kernel(const conv2d_rdna4_params p) {
#if defined(RDNA4)
    __shared__ _Float16 sA[2][BM * STRIDE];
    __shared__ _Float16 sB[2][BN * STRIDE];

    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    const int wm   = 32 * (wave / 2);  // first output channel of this wave in the block
    const int wn   = 64 * (wave % 2);  // first pixel of this wave in the block
    const int hs   = lane / 16;        // the 8-wide half of a 16-wide WMMA k step this lane holds

    const int OHW = p.OH * p.OW;
    const int p0  = blockIdx.x * BN;
    const int oc0 = blockIdx.y * BM;

    const _Float16 * xn = p.x + (int64_t) blockIdx.z * (UP ? (p.IH / 2) * (p.IW / 2) : p.IH * p.IW) * p.IC;

    // Every thread moves two 16-byte chunks of A and of B per step: rows lr and lr + 64, channels lc .. lc + 7.
    const int lr = threadIdx.x / 4;
    const int lc = 8 * (threadIdx.x % 4);
    int       oy[2], ox[2];
    bool      pv[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int pix = p0 + lr + 64 * r;
        pv[r]         = pix < OHW;
        oy[r]         = pix / p.OW;
        ox[r]         = pix % p.OW;
    }
    const int     ci_steps = p.IC / BK;
    const int     steps    = p.KH * p.KW * ci_steps;
    const int64_t w_row    = (int64_t) p.KH * p.KW * p.IC;

    uint4 ra[2], rb[2];
    auto  load = [&](const int t) {
        const int tap = t / ci_steps;
        const int ci  = (t % ci_steps) * BK + lc;
        const int ky  = tap / p.KW;
        const int kx  = tap % p.KW;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int oc = oc0 + lr + 64 * r;
            ra[r] = oc < p.OC ? *(const uint4 *) (p.w + oc * w_row + tap * p.IC + ci) : make_uint4(0, 0, 0, 0);
            const int  iy = oy[r] + ky - p.PY;
            const int  ix = ox[r] + kx - p.PX;
            const bool in = pv[r] && (unsigned) iy < (unsigned) p.IH && (unsigned) ix < (unsigned) p.IW;
            rb[r] = load_pixel<UP>(p, xn, in, iy, ix, ci);
        }
    };
    auto store = [&](const int buf) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            *(uint4 *) (sA[buf] + (lr + 64 * r) * STRIDE + lc) = ra[r];
            *(uint4 *) (sB[buf] + (lr + 64 * r) * STRIDE + lc) = rb[r];
        }
    };

    float8_t acc[2][4] = {};
    load(0);
    store(0);
    __syncthreads();
    for (int t = 0; t < steps; ++t) {
        const int buf = t & 1;
        if (t + 1 < steps) {
            load(t + 1);
        }
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            half8_t a[2], b[4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                a[i] = *(const half8_t *) (sA[buf] + (wm + 16 * i + lane % 16) * STRIDE + ks + 8 * hs);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                b[j] = *(const half8_t *) (sB[buf] + (wn + 16 * j + lane % 16) * STRIDE + ks + 8 * hs);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
                }
            }
        }
        if (t + 1 < steps) {
            store(buf ^ 1);
        }
        __syncthreads();
    }

    // Result layout: lane holds pixel lane % 16 of a 16-pixel tile and output channels 8 hs .. 8 hs + 7 of a 16-channel tile.
    float *       yn = p.y + (int64_t) blockIdx.z * p.OC * OHW;
    const float * rn = p.res != nullptr ? p.res + (int64_t) blockIdx.z * p.OC * OHW : nullptr;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int pix = p0 + wn + 16 * j + lane % 16;
        if (pix >= OHW) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int oc = oc0 + wm + 16 * i + 8 * hs + v;
                if (oc < p.OC) {
                    float r = acc[i][j][v] + (p.bias != nullptr ? p.bias[oc] : 0.0f);
                    if (rn != nullptr) {
                        r = r + rn[(int64_t) oc * OHW + pix];
                    }
                    yn[(int64_t) oc * OHW + pix] = r;
                }
            }
        }
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

// Convolutions with at most 16 output channels (the decoders' conv_out has 3): one 16-row tile of output channels
// instead of 128, NT / 32 waves of 16 x (BP * 32 / NT) pixels. The K order is conv2d_rdna4_kernel's (taps, 32 input
// channels per step, two 16-wide WMMA steps), so every output is summed the same way.
template <int NT, int BP>
__global__ void __launch_bounds__(NT) conv2d_rdna4_small_oc_kernel(const conv2d_rdna4_params p) {
#if defined(RDNA4)
    constexpr int WP = BP / (NT / 32);  // pixels per wave
    constexpr int JT = WP / 16;         // 16-pixel tiles per wave
    constexpr int RB = BP * 4 / NT;     // 16-byte B chunks per thread and step
    __shared__ _Float16 sA[2][SM_BM * STRIDE];
    __shared__ _Float16 sB[2][BP * STRIDE];

    const int lane = threadIdx.x % 32;
    const int wn   = WP * (threadIdx.x / 32);
    const int hs   = lane / 16;

    const int OHW = p.OH * p.OW;
    const int p0  = blockIdx.x * BP;

    const _Float16 * xn = p.x + (int64_t) blockIdx.z * p.IH * p.IW * p.IC;

    // B: rows lr + (NT / 4) r; A: row lr of threads 0 .. 63.
    const int lr = threadIdx.x / 4;
    const int lc = 8 * (threadIdx.x % 4);
    int       oy[RB], ox[RB];
    bool      pv[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) {
        const int pix = p0 + lr + (NT / 4) * r;
        pv[r]         = pix < OHW;
        oy[r]         = pix / p.OW;
        ox[r]         = pix % p.OW;
    }
    const int     ci_steps = p.IC / BK;
    const int     steps    = p.KH * p.KW * ci_steps;
    const int64_t w_row    = (int64_t) p.KH * p.KW * p.IC;

    uint4 ra, rb[RB];
    auto  load = [&](const int t) {
        const int tap = t / ci_steps;
        const int ci  = (t % ci_steps) * BK + lc;
        const int ky  = tap / p.KW;
        const int kx  = tap % p.KW;
        ra = lr < p.OC ? *(const uint4 *) (p.w + lr * w_row + tap * p.IC + ci) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            const int  iy = oy[r] + ky - p.PY;
            const int  ix = ox[r] + kx - p.PX;
            const bool in = pv[r] && (unsigned) iy < (unsigned) p.IH && (unsigned) ix < (unsigned) p.IW;
            rb[r] = load_pixel<false>(p, xn, in, iy, ix, ci);
        }
    };
    auto store = [&](const int buf) {
        if (lr < SM_BM) {
            *(uint4 *) (sA[buf] + lr * STRIDE + lc) = ra;
        }
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            *(uint4 *) (sB[buf] + (lr + (NT / 4) * r) * STRIDE + lc) = rb[r];
        }
    };

    float8_t acc[JT] = {};
    load(0);
    store(0);
    __syncthreads();
    for (int t = 0; t < steps; ++t) {
        const int buf = t & 1;
        if (t + 1 < steps) {
            load(t + 1);
        }
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            const half8_t a = *(const half8_t *) (sA[buf] + (lane % 16) * STRIDE + ks + 8 * hs);
            half8_t       b[JT];
#pragma unroll
            for (int j = 0; j < JT; ++j) {
                b[j] = *(const half8_t *) (sB[buf] + (wn + 16 * j + lane % 16) * STRIDE + ks + 8 * hs);
            }
#pragma unroll
            for (int j = 0; j < JT; ++j) {
                acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b[j], acc[j]);
            }
        }
        if (t + 1 < steps) {
            store(buf ^ 1);
        }
        __syncthreads();
    }

    float *       yn = p.y + (int64_t) blockIdx.z * p.OC * OHW;
    const float * rn = p.res != nullptr ? p.res + (int64_t) blockIdx.z * p.OC * OHW : nullptr;
#pragma unroll
    for (int j = 0; j < JT; ++j) {
        const int pix = p0 + wn + 16 * j + lane % 16;
        if (pix >= OHW) {
            continue;
        }
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int oc = 8 * hs + v;
            if (oc < p.OC) {
                float r = acc[j][v] + (p.bias != nullptr ? p.bias[oc] : 0.0f);
                if (rn != nullptr) {
                    r = r + rn[(int64_t) oc * OHW + pix];
                }
                yn[(int64_t) oc * OHW + pix] = r;
            }
        }
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

// [N][C][HW] f32 -> [N][HW][C] f16 through a 64 x 64 tile in LDS.
__global__ void conv2d_rdna4_to_nhwc(const float * __restrict__ x, _Float16 * __restrict__ y, const int C, const int HW) {
    __shared__ float tile[64][65];
    const int     p0  = blockIdx.x * 64;
    const int     c0  = blockIdx.y * 64;
    const int64_t off = (int64_t) blockIdx.z * C * HW;
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
        const int c = i / 64, q = i % 64;
        tile[c][q]  = c0 + c < C && p0 + q < HW ? x[off + (int64_t) (c0 + c) * HW + p0 + q] : 0.0f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
        const int q = i / 64, c = i % 64;
        if (p0 + q < HW && c0 + c < C) {
            y[off + (int64_t) (p0 + q) * C + c0 + c] = (_Float16) tile[c][q];
        }
    }
}

// [OC][IC][KH * KW] -> [OC][KH * KW][IC]
__global__ void conv2d_rdna4_reorder_weights(const _Float16 * __restrict__ w, _Float16 * __restrict__ wr, const int IC,
                                             const int taps, const int64_t total) {
    const int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int     ci  = i % IC;
    const int64_t r   = i / IC;
    const int     tap = r % taps;
    const int64_t oc  = r / taps;
    wr[i]             = w[(oc * IC + ci) * taps + tap];
}

// One consumer and not a graph output. In-place ops (the bias ADD) are views, which ggml_node_has_n_uses rejects; with
// one consumer nobody else reads the values.
bool single_consumer(const ggml_cgraph * cgraph, const int idx) {
    return ggml_node_get_use_count(cgraph, idx) == 1 && (cgraph->nodes[idx]->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
}

} // namespace

bool ggml_cuda_conv2d_rdna4_supported(const int cc, const ggml_tensor * conv) {
    const ggml_tensor * w  = conv->src[0];
    const ggml_tensor * x  = conv->src[1];
    const int32_t *     op = (const int32_t *) conv->op_params;  // stride x/y, padding x/y, dilation x/y, cwhn
    return GGML_CUDA_CC_IS_RDNA4(cc) && w->type == GGML_TYPE_F16 && x->type == GGML_TYPE_F32 &&
           conv->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_is_contiguous(x) && ggml_is_contiguous(conv) &&
           op[0] == 1 && op[1] == 1 && op[4] == 1 && op[5] == 1 && op[6] == 0 && op[2] >= 0 && op[3] >= 0 &&
           x->ne[2] % BK == 0 && x->ne[3] <= 65535 && w->ne[3] <= 65535 * BM &&
           conv->ne[0] * conv->ne[1] <= INT_MAX - BN && ggml_nelements(w) <= INT_MAX;
}

void ggml_cuda_conv2d_rdna4_to_nhwc(cudaStream_t stream, const float * x, half * y, const int C, const int HW, const int N) {
    conv2d_rdna4_to_nhwc<<<dim3((HW + 63) / 64, (C + 63) / 64, N), 256, 0, stream>>>(x, (_Float16 *) y, C, HW);
}

void ggml_cuda_conv2d_rdna4_ex(ggml_backend_cuda_context & ctx, const ggml_tensor * conv, const half * x_nhwc, const bool up2x,
                               const float * bias, const float * res, float * out) {
    const ggml_tensor * w  = conv->src[0];
    const ggml_tensor * x  = conv->src[1];
    const int32_t *     op = (const int32_t *) conv->op_params;
    const int           IW = x->ne[0], IH = x->ne[1], IC = x->ne[2], N = x->ne[3];
    const int           KW = w->ne[0], KH = w->ne[1], OC = w->ne[3];
    const int           OW = conv->ne[0], OH = conv->ne[1];
    cudaStream_t        stream = ctx.stream();
    GGML_ASSERT(!up2x || x_nhwc != nullptr);

    ggml_cuda_pool_alloc<half> x_conv(ctx.pool());
    if (x_nhwc == nullptr) {
        x_conv.alloc((size_t) N * IC * IH * IW);
        ggml_cuda_conv2d_rdna4_to_nhwc(stream, (const float *) x->data, x_conv.get(), IC, IH * IW, N);
        x_nhwc = x_conv.get();
    }
    ggml_cuda_pool_alloc<half> w_taps(ctx.pool(), ggml_nelements(w));
    const int64_t w_total = ggml_nelements(w);
    conv2d_rdna4_reorder_weights<<<(w_total + 255) / 256, 256, 0, stream>>>(
        (const _Float16 *) w->data, (_Float16 *) w_taps.get(), IC, KH * KW, w_total);

    const conv2d_rdna4_params p = { (const _Float16 *) x_nhwc, (const _Float16 *) w_taps.get(), bias, res, out,
                                    IW, IH, OW, OH, KW, KH, op[2], op[3], IC, OC };
    if (OC <= SM_BM && !up2x) {
        // 4 waves x 32 pixels per block: the fastest of 64-256 threads x 128-256 pixels for conv_out at 1024 x 1024.
        conv2d_rdna4_small_oc_kernel<128, 128><<<dim3((OH * OW + 127) / 128, 1, N), 128, 0, stream>>>(p);
    } else if (up2x) {
        conv2d_rdna4_kernel<true><<<dim3((OH * OW + BN - 1) / BN, (OC + BM - 1) / BM, N), THREADS, 0, stream>>>(p);
    } else {
        conv2d_rdna4_kernel<false><<<dim3((OH * OW + BN - 1) / BN, (OC + BM - 1) / BM, N), THREADS, 0, stream>>>(p);
    }
}

void ggml_cuda_conv2d_rdna4(ggml_backend_cuda_context & ctx, const ggml_tensor * conv, const float * bias, float * out) {
    ggml_cuda_conv2d_rdna4_ex(ctx, conv, nullptr, false, bias, nullptr, out);
}

int ggml_cuda_conv2d_rdna4_match(const ggml_cgraph * cgraph, const int i, const float ** bias, const float ** res,
                                 float ** out) {
    const ggml_tensor * conv = cgraph->nodes[i];
    if (!ggml_node_has_n_uses(cgraph, i, 1)) {
        return -1;
    }
    const int j = ggml_cuda_next_compute_node(cgraph, i);
    if (j >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * add = cgraph->nodes[j];
    if (add->op != GGML_OP_ADD || (add->src[0] != conv && add->src[1] != conv)) {
        return -1;
    }
    const ggml_tensor * b = add->src[0] == conv ? add->src[1] : add->src[0];
    if (b->type != GGML_TYPE_F32 || !ggml_is_contiguous(b) || ggml_nelements(b) != conv->ne[2] || b->ne[2] != conv->ne[2] ||
            add->type != GGML_TYPE_F32 || !ggml_are_same_shape(add, conv) || !ggml_is_contiguous(add)) {
        return -1;
    }
    *bias = (const float *) b->data;
    *res  = nullptr;
    *out  = (float *) add->data;

    // The skip of a resnet or attention block: (conv + bias) + r, summed in that order as the two ADDs would.
    const int k = ggml_cuda_next_compute_node(cgraph, j);
    if (k < cgraph->n_nodes && single_consumer(cgraph, j)) {
        const ggml_tensor * skip = cgraph->nodes[k];
        if (skip->op == GGML_OP_ADD && (skip->src[0] == add || skip->src[1] == add)) {
            const ggml_tensor * r = skip->src[0] == add ? skip->src[1] : skip->src[0];
            if (r != add && r->type == GGML_TYPE_F32 && ggml_is_contiguous(r) && ggml_are_same_shape(r, add) &&
                    skip->type == GGML_TYPE_F32 && ggml_is_contiguous(skip) && ggml_are_same_shape(skip, add)) {
                *res = (const float *) r->data;
                *out = (float *) skip->data;
                return k;
            }
        }
    }
    return j;
}

int ggml_cuda_try_fuse_upscale_conv(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, const int i) {
    const ggml_tensor * up  = cgraph->nodes[i];
    const ggml_tensor * src = up->src[0];
    if (up->op != GGML_OP_UPSCALE || ggml_get_op_params_i32(up, 0) != GGML_SCALE_MODE_NEAREST ||
            src->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || !ggml_is_contiguous(src) ||
            up->ne[0] != 2 * src->ne[0] || up->ne[1] != 2 * src->ne[1] || up->ne[2] != src->ne[2] ||
            up->ne[3] != src->ne[3] || !ggml_node_has_n_uses(cgraph, i, 1)) {
        return 0;
    }
    const int i_conv = ggml_cuda_next_compute_node(cgraph, i);
    if (i_conv >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * conv = cgraph->nodes[i_conv];
    if (conv->op != GGML_OP_CONV_2D || conv->src[1] != up ||
            !ggml_cuda_conv2d_rdna4_supported(ggml_cuda_info().devices[ctx.device].cc, conv)) {
        return 0;
    }
    const float * bias;
    const float * res;
    float *       out;
    const int     last = ggml_cuda_conv2d_rdna4_match(cgraph, i_conv, &bias, &res, &out);
    if (last < 0) {
        return 0;
    }
    const int                  C = src->ne[2], HW = src->ne[0] * src->ne[1], N = src->ne[3];
    ggml_cuda_pool_alloc<half> x_nhwc(ctx.pool(), (size_t) N * C * HW);
    ggml_cuda_conv2d_rdna4_to_nhwc(ctx.stream(), (const float *) src->data, x_nhwc.get(), C, HW, N);
    ggml_cuda_conv2d_rdna4_ex(ctx, conv, x_nhwc.get(), true, bias, res, out);
    return last - i;
}
