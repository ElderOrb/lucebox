#include "group-norm-fused.cuh"
#include "conv2d-rdna4.cuh"

// In ggml's [W, H, C, N] layout the C / G channels of a group are one contiguous range of values. The stats pass
// splits every group into chunks of CHUNK values; each block leaves Welford partials (count, mean, M2) that the
// normalize pass merges (Chan et al.) before writing y = (x - mean) * rstd * w[c] + b[c], then silu(y). The generic
// kernel runs one block per group, which on the 1024 x 1024 x 128 maps of a VAE decoder reads at ~200 GB/s; here every
// pass runs at full bandwidth.

namespace {

constexpr int CHUNK   = 16384;  // values per block
constexpr int THREADS = 256;

struct welford {
    float n, mean, m2;
};

__device__ __forceinline__ welford welford_merge(const welford a, const welford b) {
    const float n = a.n + b.n;
    if (n == 0.0f) {
        return a;
    }
    const float d = b.mean - a.mean;
    const float f = b.n / n;
    return { n, a.mean + d * f, a.m2 + b.m2 + d * d * a.n * f };
}

__device__ __forceinline__ welford welford_warp_reduce(welford w) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
#pragma unroll
    for (int off = warp_size / 2; off > 0; off >>= 1) {
        const welford o = { __shfl_xor_sync(0xFFFFFFFF, w.n, off, warp_size), __shfl_xor_sync(0xFFFFFFFF, w.mean, off, warp_size),
                            __shfl_xor_sync(0xFFFFFFFF, w.m2, off, warp_size) };
        w = welford_merge(w, o);
    }
    return w;
}

__global__ void __launch_bounds__(THREADS) group_norm_stats(const float * __restrict__ x, welford * __restrict__ partial,
                                                            const int64_t group_size) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int64_t group = (int64_t) blockIdx.z * gridDim.y + blockIdx.y;
    const float * xg    = x + group * group_size;
    const int64_t end   = min(group_size, (int64_t) (blockIdx.x + 1) * CHUNK);
    welford       w     = { 0.0f, 0.0f, 0.0f };
    for (int64_t k = (int64_t) blockIdx.x * CHUNK + 4 * threadIdx.x; k < end; k += 4 * THREADS) {
        const float4 v = *(const float4 *) (xg + k);
        for (const float e : { v.x, v.y, v.z, v.w }) {
            w.n += 1.0f;
            const float d = e - w.mean;
            w.mean += d / w.n;
            w.m2 += d * (e - w.mean);
        }
    }
    w = welford_warp_reduce(w);
    __shared__ welford s[THREADS / warp_size];
    if (threadIdx.x % warp_size == 0) {
        s[threadIdx.x / warp_size] = w;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < THREADS / warp_size; ++i) {
            w = welford_merge(w, s[i]);
        }
        partial[group * gridDim.x + blockIdx.x] = w;
    }
}

template <bool silu>
__global__ void __launch_bounds__(THREADS) group_norm_apply(const float * __restrict__ x, const welford * __restrict__ partial,
                                                            const float * __restrict__ w, const float * __restrict__ b,
                                                            float * y, const int64_t group_size, const int64_t hw,
                                                            const int channels_per_group, const float eps) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int64_t group = (int64_t) blockIdx.z * gridDim.y + blockIdx.y;
    __shared__ float s_mean, s_rstd;
    if (threadIdx.x < warp_size) {
        welford acc = { 0.0f, 0.0f, 0.0f };
        for (int c = threadIdx.x; c < (int) gridDim.x; c += warp_size) {
            acc = welford_merge(acc, partial[group * gridDim.x + c]);
        }
        acc = welford_warp_reduce(acc);
        if (threadIdx.x == 0) {
            s_mean = acc.mean;
            s_rstd = rsqrtf(acc.m2 / acc.n + eps);
        }
    }
    __syncthreads();
    const float   mean = s_mean;
    const float   rstd = s_rstd;
    const int     c0   = blockIdx.y * channels_per_group;
    const float * xg   = x + group * group_size;
    float *       yg   = y + group * group_size;
    const int64_t end  = min(group_size, (int64_t) (blockIdx.x + 1) * CHUNK);
    for (int64_t k = (int64_t) blockIdx.x * CHUNK + 4 * threadIdx.x; k < end; k += 4 * THREADS) {
        const int    c  = c0 + (int) (k / hw);  // hw % 4 == 0: the 4 values share a channel
        const float  sc = rstd * w[c];
        const float  bc = b[c];
        const float4 v  = *(const float4 *) (xg + k);
        float        o[4] = { (v.x - mean) * sc + bc, (v.y - mean) * sc + bc, (v.z - mean) * sc + bc, (v.w - mean) * sc + bc };
        if constexpr (silu) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                o[i] = o[i] / (1.0f + expf(-o[i]));
            }
        }
        *(float4 *) (yg + k) = make_float4(o[0], o[1], o[2], o[3]);
    }
}

// The mean and 1 / std of every group, merged from the stats partials exactly as group_norm_apply merges them.
__global__ void group_norm_finalize(const welford * __restrict__ partial, float2 * __restrict__ stats, const int chunks,
                                    const float eps) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int64_t group = (int64_t) blockIdx.z * gridDim.y + blockIdx.y;
    welford       acc   = { 0.0f, 0.0f, 0.0f };
    for (int c = threadIdx.x; c < chunks; c += warp_size) {
        acc = welford_merge(acc, partial[group * chunks + c]);
    }
    acc = welford_warp_reduce(acc);
    if (threadIdx.x == 0) {
        stats[group] = make_float2(acc.mean, rsqrtf(acc.m2 / acc.n + eps));
    }
}

// The normalize pass for a convolution that is the map's only consumer: group_norm_apply's arithmetic, rounded to f16
// and written channels-last (the convolution's input layout) through a 64 x 64 tile in LDS. A tile word holds the
// channel pair (2 k, 2 k + 1) of one pixel; 33 words per pixel keep both passes free of bank conflicts.
template <bool silu>
__global__ void __launch_bounds__(THREADS) group_norm_apply_nhwc(const float * __restrict__ x, const float2 * __restrict__ stats,
                                                                 const float * __restrict__ w, const float * __restrict__ b,
                                                                 _Float16 * __restrict__ y, const int C, const int64_t hw,
                                                                 const int channels_per_group, const int groups) {
    __shared__ uint32_t tile[64][33];
    const int64_t p0 = (int64_t) blockIdx.x * 64;
    const int     c0 = blockIdx.y * 64;
    const int     n  = blockIdx.z;
    const int     q  = 4 * (threadIdx.x % 16);
    for (int k = threadIdx.x / 16; k < 32; k += THREADS / 16) {
        float o[2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int    c    = c0 + 2 * k + h;
            const float2 st   = stats[(int64_t) n * groups + c / channels_per_group];
            const float  mean = st.x;
            const float  rstd = st.y;
            const float  sc   = rstd * w[c];
            const float  bc   = b[c];
            const float4 v    = *(const float4 *) (x + ((int64_t) n * C + c) * hw + p0 + q);
            o[h][0] = (v.x - mean) * sc + bc;
            o[h][1] = (v.y - mean) * sc + bc;
            o[h][2] = (v.z - mean) * sc + bc;
            o[h][3] = (v.w - mean) * sc + bc;
            if constexpr (silu) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    o[h][i] = o[h][i] / (1.0f + expf(-o[h][i]));
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const _Float16 lo = (_Float16) o[0][i];
            const _Float16 hi = (_Float16) o[1][i];
            tile[q + i][k] = (uint32_t) __builtin_bit_cast(uint16_t, lo) | ((uint32_t) __builtin_bit_cast(uint16_t, hi) << 16);
        }
    }
    __syncthreads();
    const int k0 = 4 * (threadIdx.x % 8);  // channels 2 k0 .. 2 k0 + 7
    for (int qq = threadIdx.x / 8; qq < 64; qq += THREADS / 8) {
        const uint4 v = make_uint4(tile[qq][k0], tile[qq][k0 + 1], tile[qq][k0 + 2], tile[qq][k0 + 3]);
        *(uint4 *) (y + ((int64_t) n * hw + p0 + qq) * C + c0 + 2 * k0) = v;
    }
}

// A [1, 1, C, 1] f32 per-channel operand of node, or null.
const float * channel_operand(const ggml_tensor * node, const ggml_tensor * prev, const int64_t C) {
    const ggml_tensor * t = node->src[0] == prev ? node->src[1] : node->src[0];
    if (t == nullptr || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t) != C || t->ne[2] != C) {
        return nullptr;
    }
    return (const float *) t->data;
}

// One consumer and not a graph output. The in-place MUL, ADD and SILU are views of the norm's buffer, which the
// generic single-use check rejects; with one consumer each nobody else reads the intermediate values.
bool single_consumer(const ggml_cgraph * cgraph, const int idx) {
    return ggml_node_get_use_count(cgraph, idx) == 1 && (cgraph->nodes[idx]->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
}

} // namespace

int ggml_cuda_next_compute_node(const ggml_cgraph * cgraph, int j) {
    for (++j; j < cgraph->n_nodes && ggml_op_is_empty(cgraph->nodes[j]->op); ++j) {
    }
    return j;
}

int ggml_cuda_try_fuse_group_norm(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, const int i) {
    const ggml_tensor * gn = cgraph->nodes[i];
    const ggml_tensor * x  = gn->src[0];
    if (gn->op != GGML_OP_GROUP_NORM || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(gn) ||
            !single_consumer(cgraph, i)) {
        return 0;
    }
    const int     groups = ggml_get_op_params_i32(gn, 0);
    const float   eps    = ggml_get_op_params_f32(gn, 1);
    const int64_t C      = x->ne[2];
    const int64_t hw     = x->ne[0] * x->ne[1];
    if (groups <= 0 || C % groups != 0 || hw % 4 != 0 || x->ne[3] > 65535 || groups > 65535) {
        return 0;
    }
    const int i_mul = ggml_cuda_next_compute_node(cgraph, i);
    const int i_add = ggml_cuda_next_compute_node(cgraph, i_mul);
    if (i_add >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * mul = cgraph->nodes[i_mul];
    const ggml_tensor * add = cgraph->nodes[i_add];
    if (mul->op != GGML_OP_MUL || (mul->src[0] != gn && mul->src[1] != gn) || !single_consumer(cgraph, i_mul) ||
            add->op != GGML_OP_ADD || (add->src[0] != mul && add->src[1] != mul) || !ggml_are_same_shape(add, gn)) {
        return 0;
    }
    const float * w = channel_operand(mul, gn, C);
    const float * b = channel_operand(add, mul, C);
    if (w == nullptr || b == nullptr) {
        return 0;
    }
    int                 last = i_add;
    const ggml_tensor * out  = add;
    const int           i_silu = ggml_cuda_next_compute_node(cgraph, i_add);
    if (i_silu < cgraph->n_nodes && single_consumer(cgraph, i_add)) {
        const ggml_tensor * silu = cgraph->nodes[i_silu];
        if (silu->op == GGML_OP_UNARY && ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU && silu->src[0] == add &&
                ggml_are_same_shape(silu, add) && ggml_is_contiguous(silu)) {
            last = i_silu;
            out  = silu;
        }
    }
    if (out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out)) {
        return 0;
    }

    const int64_t group_size = (C / groups) * hw;
    const int     chunks     = (int) ((group_size + CHUNK - 1) / CHUNK);
    const dim3    grid(chunks, groups, x->ne[3]);
    cudaStream_t  stream = ctx.stream();

    // A single RDNA4 convolution reading the normalized map (the resnet convolutions and conv_out of the image VAEs):
    // write the map straight in the convolution's f16 channels-last input layout and run the convolution with its bias
    // and skip ADD. The f32 map is never written.
    const int i_conv = ggml_cuda_next_compute_node(cgraph, last);
    if (i_conv < cgraph->n_nodes && hw % 64 == 0 && C % 64 == 0 && single_consumer(cgraph, last)) {
        const ggml_tensor * conv = cgraph->nodes[i_conv];
        const float *       bias;
        const float *       res;
        float *             y;
        int                 end = -1;
        if (conv->op == GGML_OP_CONV_2D && conv->src[1] == out &&
                ggml_cuda_conv2d_rdna4_supported(ggml_cuda_info().devices[ctx.device].cc, conv)) {
            end = ggml_cuda_conv2d_rdna4_match(cgraph, i_conv, &bias, &res, &y);
        }
        if (end >= 0) {
            const int                     warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
            ggml_cuda_pool_alloc<welford> partial(ctx.pool(), (size_t) chunks * groups * x->ne[3]);
            ggml_cuda_pool_alloc<float2>  stats(ctx.pool(), (size_t) groups * x->ne[3]);
            ggml_cuda_pool_alloc<half>    x_nhwc(ctx.pool(), (size_t) x->ne[3] * C * hw);
            group_norm_stats<<<grid, THREADS, 0, stream>>>((const float *) x->data, partial.get(), group_size);
            group_norm_finalize<<<dim3(1, groups, x->ne[3]), warp_size, 0, stream>>>(partial.get(), stats.get(), chunks, eps);
            const dim3 grid_nhwc(hw / 64, C / 64, x->ne[3]);
            if (out != add) {
                group_norm_apply_nhwc<true><<<grid_nhwc, THREADS, 0, stream>>>((const float *) x->data, stats.get(), w, b,
                    (_Float16 *) x_nhwc.get(), (int) C, hw, (int) (C / groups), groups);
            } else {
                group_norm_apply_nhwc<false><<<grid_nhwc, THREADS, 0, stream>>>((const float *) x->data, stats.get(), w, b,
                    (_Float16 *) x_nhwc.get(), (int) C, hw, (int) (C / groups), groups);
            }
            ggml_cuda_conv2d_rdna4_ex(ctx, conv, x_nhwc.get(), false, bias, res, y);
            return end - i;
        }
    }

    ggml_cuda_pool_alloc<welford> partial(ctx.pool(), (size_t) chunks * groups * x->ne[3]);
    group_norm_stats<<<grid, THREADS, 0, stream>>>((const float *) x->data, partial.get(), group_size);
    if (out != add) {
        group_norm_apply<true><<<grid, THREADS, 0, stream>>>((const float *) x->data, partial.get(), w, b, (float *) out->data,
                                                             group_size, hw, (int) (C / groups), eps);
    } else {
        group_norm_apply<false><<<grid, THREADS, 0, stream>>>((const float *) x->data, partial.get(), w, b, (float *) out->data,
                                                              group_size, hw, (int) (C / groups), eps);
    }
    return last - i;
}
