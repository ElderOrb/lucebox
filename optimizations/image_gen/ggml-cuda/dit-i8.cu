#include "dit-i8.cuh"
#include "gemm-i8.cuh"
#include "convrot.cuh"
#include "convrot-wave.cuh"
#include "rope.cuh"
#include "unary.cuh"

static __device__ __forceinline__ float dequantize_i32(
        const int32_t sum, const float act_scale, const float * weight_scales, const float * bias, const int64_t col) {
    float value = (float) sum * act_scale;
    if (weight_scales != nullptr) {
        value *= weight_scales[col];
    }
    if (bias != nullptr) {
        value += bias[col];
    }
    return value;
}

static __device__ __forceinline__ float block_reduce_sum_1024(float value, float * partial) {
    value = warp_reduce_sum<32>(value);
    if ((threadIdx.x & 31) == 0) {
        partial[threadIdx.x >> 5] = value;
    }
    __syncthreads();
    value = threadIdx.x < 32 ? partial[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
        value = warp_reduce_sum<32>(value);
        if (threadIdx.x == 0) {
            partial[0] = value;
        }
    }
    __syncthreads();
    value = partial[0];
    __syncthreads();
    return value;
}

// One block of 1024 threads per row; the row lives in shared memory between the norm and the quantization.
static __global__ void rms_norm_mul_add_quantize_i8_convrot_kernel(
        const float * x, const float * w, const float * b, const float eps, const int64_t k, int8_t * dst, float * scales) {
    extern __shared__ float values[];
    __shared__ float partial[32];
    const int64_t row   = blockIdx.x;
    const float * x_row = x + row * k;

    float sumsq = 0.0f;
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        const float v = x_row[i];
        values[i] = v;
        sumsq += v * v;
    }
    sumsq = block_reduce_sum_1024(sumsq, partial);
    const float scale = rsqrtf(sumsq / k + eps);
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        values[i] = (scale * values[i] * w[i] + b[i]) * (1.0f / 16.0f);
    }
    __syncthreads();
    ggml_cuda_convrot_quantize_row_i8(values, k, dst + row * k, scales + row, partial);
}

void ggml_cuda_rms_norm_mul_add_quantize_i8_convrot(
        const float * x, const float * w, const float * b, const float eps, const int64_t k, const int64_t rows,
        int8_t * dst, float * scales, cudaStream_t stream) {
    if (ggml_cuda_convrot_wave(x, w, b, eps, k, rows, dst, k, scales, stream)) {
        return;
    }
    const size_t nbytes_shared = (size_t) k * sizeof(float);
    CUDA_SET_SHARED_MEMORY_LIMIT(rms_norm_mul_add_quantize_i8_convrot_kernel, ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo - 128);
    rms_norm_mul_add_quantize_i8_convrot_kernel<<<rows, 1024, nbytes_shared, stream>>>(x, w, b, eps, k, dst, scales);
}

// Four outputs per thread.
static __global__ void situ_i32_kernel(
        const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
        const int64_t nc, const int64_t total4, const int64_t gate_offset, const int64_t up_offset,
        const float gate_cap, const float up_cap, float4 * dst) {
    const int64_t i4 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i4 >= total4) {
        return;
    }
    const int64_t i   = 4 * i4;
    const int64_t row = i / nc;
    const int64_t col = i - row * nc;
    const int32_t * row_sums = sums + row * 2 * nc;
    const float act_scale    = act_scales[row];
    float out[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float gate = dequantize_i32(row_sums[gate_offset + col + j], act_scale, weight_scales, bias, gate_offset + col + j);
        const float up   = dequantize_i32(row_sums[up_offset + col + j], act_scale, weight_scales, bias, up_offset + col + j);
        out[j] = ggml_cuda_op_situ_single(gate, up, gate_cap, up_cap);
    }
    dst[i4] = make_float4(out[0], out[1], out[2], out[3]);
}

void ggml_cuda_situ_i32(
        const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
        const int64_t nc, const int64_t rows, const bool swapped, const float gate_cap, const float up_cap, float * dst,
        cudaStream_t stream) {
    GGML_ASSERT(nc % 4 == 0);
    const int64_t total4 = nc * rows / 4;
    situ_i32_kernel<<<(total4 + 255) / 256, 256, 0, stream>>>(sums, act_scales, weight_scales, bias, nc, total4,
                                                              swapped ? nc : 0, swapped ? 0 : nc, gate_cap, up_cap,
                                                              (float4 *) dst);
}

static __global__ void situ_i32_quantize_i8_convrot_kernel(
        const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
        const int64_t nc, const int64_t gate_offset, const int64_t up_offset, const float gate_cap, const float up_cap,
        int8_t * dst, float * scales) {
    extern __shared__ float values[];
    __shared__ float maxima[32];
    const int64_t row        = blockIdx.x;
    const int32_t * row_sums = sums + row * 2 * nc;
    const float act_scale    = act_scales[row];
    for (int64_t col4 = threadIdx.x; col4 < nc / 4; col4 += blockDim.x) {
        const int64_t col = 4 * col4;
        const int4 g = *(const int4 *) (row_sums + gate_offset + col);
        const int4 u = *(const int4 *) (row_sums + up_offset + col);
        const int32_t gs[4] = { g.x, g.y, g.z, g.w };
        const int32_t us[4] = { u.x, u.y, u.z, u.w };
        float out[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float gate = dequantize_i32(gs[j], act_scale, weight_scales, bias, gate_offset + col + j);
            const float up   = dequantize_i32(us[j], act_scale, weight_scales, bias, up_offset + col + j);
            out[j] = ggml_cuda_op_situ_single(gate, up, gate_cap, up_cap) * (1.0f / 16.0f);
        }
        ((float4 *) values)[col4] = make_float4(out[0], out[1], out[2], out[3]);
    }
    __syncthreads();
    ggml_cuda_convrot_quantize_row_i8(values, nc, dst + row * nc, scales + row, maxima);
}

// The FFN gate (situ on the dequantized int32 sums of the fused up/gate linear) quantized with convrot for the down
// linear, on RDNA2+: 8 waves per row, each walking its G 256-groups one at a time and keeping the rotated values in
// registers until the row's absolute max is known, instead of the row staged in shared memory with a block barrier per
// butterfly stage. Same float operations in the same order, so the int8 values and scales are identical.
//
// In a group, lane l holds elements i + 4 (l % 4) + 16 (l / 4 % 4) + 64 (j + 2 (l / 16)), i = 0..3, j = 0..1: the
// base-4 digits are (i, l % 4, l / 4 % 4, j + 2 (l / 16)), so the stride-1 butterfly is in-lane, stride 4 and 16 are
// DPP row_xmask exchanges (convrot-wave.cuh), and stride 64 pairs each lane with lane l ^ 16 (permlanex16).
template <int G>
static __global__ void __launch_bounds__(256) situ_i32_quantize_i8_convrot_wave_kernel(
        const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
        const int64_t gate_offset, const int64_t up_offset, const float gate_cap, const float up_cap, int8_t * dst,
        float * scales) {
#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4))
    constexpr int W  = 8;            // waves per row
    constexpr int NC = 256 * G * W;  // output values per row
    __shared__ float maxima[W];
    const int     lane        = threadIdx.x % 32;
    const int     wave        = threadIdx.x / 32;
    const int     hi          = lane / 16;
    const int     lane_offset = 4 * (lane % 16) + 128 * hi;  // element (i, j) = (0, 0) of this lane in a group
    const int64_t row         = blockIdx.x;
    const int32_t * row_sums  = sums + row * 2 * NC;
    const float     act_scale = act_scales[row];
    const auto swap16 = [](const float x) {
        return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, x), __builtin_bit_cast(int, x),
                                                                      0x76543210, 0xfedcba98, false, false));
    };

    float keep[G][2][4];
    float amax = 0.0f;
#pragma unroll 1
    for (int g = 0; g < G; ++g) {
        const int64_t base = (int64_t) (wave * G + g) * 256 + lane_offset;
        float x[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int64_t col = base + 64 * j;
            const int4 gs = *(const int4 *) (row_sums + gate_offset + col);
            const int4 us = *(const int4 *) (row_sums + up_offset + col);
            float4 gw = make_float4(1.0f, 1.0f, 1.0f, 1.0f), uw = gw;
            float4 gb = make_float4(0.0f, 0.0f, 0.0f, 0.0f), ub = gb;
            if (weight_scales != nullptr) {
                gw = *(const float4 *) (weight_scales + gate_offset + col);
                uw = *(const float4 *) (weight_scales + up_offset + col);
            }
            if (bias != nullptr) {
                gb = *(const float4 *) (bias + gate_offset + col);
                ub = *(const float4 *) (bias + up_offset + col);
            }
            const int32_t gsv[4] = { gs.x, gs.y, gs.z, gs.w };
            const int32_t usv[4] = { us.x, us.y, us.z, us.w };
            const float   gwv[4] = { gw.x, gw.y, gw.z, gw.w };
            const float   uwv[4] = { uw.x, uw.y, uw.z, uw.w };
            const float   gbv[4] = { gb.x, gb.y, gb.z, gb.w };
            const float   ubv[4] = { ub.x, ub.y, ub.z, ub.w };
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // dequantize_i32 with the scale and bias loaded as vectors
                float gate = (float) gsv[i] * act_scale;
                float up   = (float) usv[i] * act_scale;
                if (weight_scales != nullptr) {
                    gate *= gwv[i];
                    up   *= uwv[i];
                }
                if (bias != nullptr) {
                    gate += gbv[i];
                    up   += ubv[i];
                }
                x[j][i] = ggml_cuda_op_situ_single(gate, up, gate_cap, up_cap) * (1.0f / 16.0f);
            }
        }
        // strides 1, 4 and 16
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            convrot_wave_r4(x[j][0], x[j][1], x[j][2], x[j][3]);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                x[j][i] = convrot_wave_r4_lanes<1>(x[j][i], lane % 4);
            }
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                x[j][i] = convrot_wave_r4_lanes<4>(x[j][i], lane / 4 % 4);
            }
        }
        // stride 64: digits 0, 1 (a, b) are the low half-wave's j, digits 2, 3 (c, d) the high one's; output digit r
        // takes the minus sign on digit 3 - r, as in ((a + b) + c) - d for r = 0
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o0 = swap16(x[0][i]);
            const float o1 = swap16(x[1][i]);
            const float a  = hi ? o0 : x[0][i];
            const float b  = hi ? o1 : x[1][i];
            const float c  = hi ? x[0][i] : o0;
            const float d  = hi ? x[1][i] : o1;
            x[0][i] = hi ? ((a - b) + c) + d : ((a + b) + c) - d;
            x[1][i] = hi ? ((-a + b) + c) + d : ((a + b) - c) + d;
        }
        // keep the groups in order with static register indices: shift down one slot, this group last
#pragma unroll
        for (int k = 0; k + 1 < G; ++k) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    keep[k][j][i] = keep[k + 1][j][i];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                keep[G - 1][j][i] = x[j][i];
                amax = fmaxf(amax, fabsf(x[j][i]));
            }
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    if (lane == 0) {
        maxima[wave] = amax;
    }
    __syncthreads();
#pragma unroll
    for (int w = 0; w < W; ++w) {
        amax = fmaxf(amax, maxima[w]);
    }
    const float row_scale = amax / 127.0f;
    if (threadIdx.x == 0) {
        scales[row] = row_scale;
    }
    const float inv_scale = row_scale == 0.0f ? 0.0f : 1.0f / row_scale;
    const auto  quantize  = [&](const float v) {
        const int q = row_scale == 0.0f ? 0 : __float2int_rn(v * inv_scale);
        return (uint32_t) (uint8_t) (int8_t) max(-127, min(127, q));
    };
    int8_t * dst_row = dst + row * NC;
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int64_t base = (int64_t) (wave * G + g) * 256 + lane_offset;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            *(uint32_t *) (dst_row + base + 64 * j) = quantize(keep[g][j][0]) | (quantize(keep[g][j][1]) << 8) |
                                                       (quantize(keep[g][j][2]) << 16) | (quantize(keep[g][j][3]) << 24);
        }
    }
#else
    GGML_UNUSED_VARS(sums, act_scales, weight_scales, bias, gate_offset, up_offset, gate_cap, up_cap, dst, scales);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4))
}

void ggml_cuda_situ_i32_quantize_i8_convrot(
        const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
        const int64_t nc, const int64_t rows, const bool swapped, const float gate_cap, const float up_cap,
        int8_t * dst, float * scales, cudaStream_t stream) {
    const int  cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const auto aligned = [](const void * p) { return p == nullptr || (uintptr_t) p % 16 == 0; };
    if ((GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) && aligned(sums) &&
            aligned(weight_scales) && aligned(bias) && (nc == 4096 || nc == 8192)) {
        const auto kernel = nc == 4096 ? situ_i32_quantize_i8_convrot_wave_kernel<2> : situ_i32_quantize_i8_convrot_wave_kernel<4>;
        kernel<<<rows, 256, 0, stream>>>(sums, act_scales, weight_scales, bias, swapped ? nc : 0, swapped ? 0 : nc,
                                         gate_cap, up_cap, dst, scales);
        return;
    }
    const size_t nbytes_shared = (size_t) nc * sizeof(float);
    CUDA_SET_SHARED_MEMORY_LIMIT(situ_i32_quantize_i8_convrot_kernel, ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo - 128);
    situ_i32_quantize_i8_convrot_kernel<<<rows, 1024, nbytes_shared, stream>>>(
        sums, act_scales, weight_scales, bias, nc, swapped ? nc : 0, swapped ? 0 : nc, gate_cap, up_cap, dst, scales);
}

// One wave per (head, token): q and k rows normed and rotated, the v row cast.
template <bool interleaved>
static __global__ void qkv_i32_attn_prologue_kernel(
        const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
        const float * q_w, const float * k_w, const float q_eps, const float k_eps, const float * pe,
        const int64_t pe_s0, const int64_t pe_s1, const int64_t pe_s2, const int64_t pe_s3,
        const int64_t head_dim, const int64_t heads, const int64_t rows, float * q, half * k, half * v) {
    // Neighbouring waves take the heads of one token, so the sums stream through memory once (by head, they were
    // read in one strided pass per head).
    const int64_t wave_index = (int64_t) blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (wave_index >= heads * rows) {
        return;
    }
    const int64_t head      = wave_index % heads;
    const int64_t token     = wave_index / heads;
    const int64_t row       = head * rows + token;  // output row
    const int64_t dim       = heads * head_dim;
    const int32_t * s       = sums + token * 3 * dim;
    const float     act     = act_scales[token];
    const int64_t   q0      = head * head_dim;
    const int64_t   k0      = dim + q0;
    const int64_t   v0      = 2 * dim + q0;
    const float *   pe_token = pe + token * pe_s3;
    const auto load_q = [&](const int64_t i) { return dequantize_i32(s[q0 + i], act, weight_scales, bias, q0 + i); };
    const auto load_k = [&](const int64_t i) { return dequantize_i32(s[k0 + i], act, weight_scales, bias, k0 + i); };
    ggml_cuda_rms_norm_mul_rope_row<interleaved, float>(load_q, q_w, pe_token, pe_s0, pe_s1, pe_s2, head_dim / 2, q_eps,
                                                        q + row * head_dim);
    ggml_cuda_rms_norm_mul_rope_row<interleaved, half>(load_k, k_w, pe_token, pe_s0, pe_s1, pe_s2, head_dim / 2, k_eps,
                                                       k + row * head_dim);
    for (int64_t i = threadIdx.x % WARP_SIZE; i < head_dim; i += WARP_SIZE) {
        v[row * head_dim + i] = ggml_cuda_cast<half>(dequantize_i32(s[v0 + i], act, weight_scales, bias, v0 + i));
    }
}

void ggml_cuda_qkv_i32_attn_prologue(
        const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
        const float * q_w, const float * k_w, const float q_eps, const float k_eps, const float * pe,
        const int64_t pe_s0, const int64_t pe_s1, const int64_t pe_s2, const int64_t pe_s3, const bool interleaved,
        const int64_t head_dim, const int64_t heads, const int64_t rows, float * q, half * k, half * v, cudaStream_t stream) {
    const int     waves_per_block = 8;
    const int64_t blocks          = (heads * rows + waves_per_block - 1) / waves_per_block;
    const auto    kernel          = interleaved ? qkv_i32_attn_prologue_kernel<true> : qkv_i32_attn_prologue_kernel<false>;
    kernel<<<blocks, waves_per_block * WARP_SIZE, 0, stream>>>(sums, act_scales, weight_scales, bias, q_w, k_w, q_eps, k_eps,
                                                               pe, pe_s0, pe_s1, pe_s2, pe_s3, head_dim, heads, rows, q, k, v);
}

// Four outputs per thread; keeps the unfused rounding (a separate multiply and add).
static __global__ void mul_add_i32_kernel(
        const int4 * sums, const float * act_scales, const float * weight_scales, const float * bias,
        const float4 * gate, const float4 * x, float4 * dst, const int64_t n4, const int64_t total4) {
    const int64_t i4 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i4 >= total4) {
        return;
    }
    const int64_t row  = i4 / n4;
    const int64_t col4 = i4 - row * n4;
    const int4   s     = sums[i4];
    const float  a     = act_scales[row];
    const float4 g     = gate[col4];
    const float4 r     = x[i4];
    const int64_t col  = 4 * col4;
    dst[i4] = make_float4(__fadd_rn(r.x, __fmul_rn(dequantize_i32(s.x, a, weight_scales, bias, col + 0), g.x)),
                          __fadd_rn(r.y, __fmul_rn(dequantize_i32(s.y, a, weight_scales, bias, col + 1), g.y)),
                          __fadd_rn(r.z, __fmul_rn(dequantize_i32(s.z, a, weight_scales, bias, col + 2), g.z)),
                          __fadd_rn(r.w, __fmul_rn(dequantize_i32(s.w, a, weight_scales, bias, col + 3), g.w)));
}

void ggml_cuda_mul_add_i32(
        const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
        const float * gate, const float * x, float * dst, const int64_t n, const int64_t rows, cudaStream_t stream) {
    GGML_ASSERT(n % 4 == 0);
    const int64_t total4 = n * rows / 4;
    mul_add_i32_kernel<<<(total4 + 255) / 256, 256, 0, stream>>>((const int4 *) sums, act_scales, weight_scales, bias,
                                                                 (const float4 *) gate, (const float4 *) x, (float4 *) dst,
                                                                 n / 4, total4);
}

// MUL_MAT(int8) -> MUL(gate) -> ADD(x) -> RMS_NORM -> MUL(w) -> ADD(b) -> QUANTIZE_I8_CONVROT on RDNA2+: a gated
// residual whose result the next int8 linear reads through a modulated norm. Two waves per row (convrot_wave_row) write
// the new residual row and quantize it from their registers, instead of the elementwise residual kernel and a norm
// kernel that reads the row back. Same float operations, so the residual, the int8 values and scales are identical.
template <int P>
static __global__ void __launch_bounds__(64) mul_add_i32_rms_norm_quantize_i8_convrot_kernel(
        const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
        const float * gate, const float * x, float * dst, const float * w, const float * b, const float eps,
        int8_t * q_dst, float * q_scales) {
#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4))
    constexpr int K   = 512 * P;
    const int64_t row = blockIdx.x;
    const float   a   = act_scales[row];
    convrot_wave_row<P, 2, true>([&](const int p, const int q) {
        // mul_add_i32_kernel's arithmetic
        const int64_t col = convrot_wave_index(p, q);
        const int4    s   = *(const int4 *) (sums + row * K + col);
        const float4  r   = *(const float4 *) (x + row * K + col);
        const float4  g   = *(const float4 *) (gate + col);
        const float4  o   = make_float4(__fadd_rn(r.x, __fmul_rn(dequantize_i32(s.x, a, weight_scales, bias, col + 0), g.x)),
                                        __fadd_rn(r.y, __fmul_rn(dequantize_i32(s.y, a, weight_scales, bias, col + 1), g.y)),
                                        __fadd_rn(r.z, __fmul_rn(dequantize_i32(s.z, a, weight_scales, bias, col + 2), g.z)),
                                        __fadd_rn(r.w, __fmul_rn(dequantize_i32(s.w, a, weight_scales, bias, col + 3), g.w)));
        *(float4 *) (dst + row * K + col) = o;
        return o;
    }, w, b, eps, q_dst + row * K, q_scales + row);
#else
    GGML_UNUSED_VARS(sums, act_scales, weight_scales, bias, gate, x, dst, w, b, eps, q_dst, q_scales);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4))
}

bool ggml_cuda_mul_add_i32_rms_norm_quantize_i8_convrot(
        const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
        const float * gate, const float * x, float * dst, const int64_t n, const int64_t rows,
        const float * w, const float * b, const float eps, int8_t * q_dst, float * q_scales, cudaStream_t stream) {
    const int  cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const auto aligned = [](const void * p) { return (uintptr_t) p % 16 == 0; };
    if (!(GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) || !aligned(sums) ||
            !aligned(gate) || !aligned(x) || !aligned(dst) || !aligned(w) || !aligned(b) || (uintptr_t) q_dst % 4 != 0) {
        return false;
    }
    const auto launch = [&](auto kernel) {
        kernel<<<rows, 64, 0, stream>>>(sums, act_scales, weight_scales, bias, gate, x, dst, w, b, eps, q_dst, q_scales);
        return true;
    };
    switch (n) {
        case 2048: return launch(mul_add_i32_rms_norm_quantize_i8_convrot_kernel<4>);
        case 3072: return launch(mul_add_i32_rms_norm_quantize_i8_convrot_kernel<6>);
        case 4096: return launch(mul_add_i32_rms_norm_quantize_i8_convrot_kernel<8>);
        default:   return false;
    }
}

// Graph matching: the node chains of a DiT block that the kernels above and the fused GEMMs of gemm-i8.cu replace.

bool ggml_cuda_can_fuse_chain(const ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    if (node_idx + (int) ops.size() > cgraph->n_nodes) {
        return false;
    }
    int i = 0;
    for (const enum ggml_op op : ops) {
        const ggml_tensor * node = cgraph->nodes[node_idx + i];
        if (node->op != op || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
        if (i < (int) ops.size() - 1 && !ggml_node_has_n_uses(cgraph, node_idx + i, 1)) {
            return false;
        }
        if (i > 0 && node->src[0] != cgraph->nodes[node_idx + i - 1] && node->src[1] != cgraph->nodes[node_idx + i - 1]) {
            return false;
        }
        ++i;
    }
    return true;
}

#if !defined(GGML_USE_MUSA)
// A QUANTIZE_I8_CONVROT output for rows of k values: groups of 256, int8 rows padded to 4, then the row scales; the
// row must fit in shared memory for the fused row kernels.
static bool ggml_cuda_i8_convrot_output_ok(const ggml_tensor * q, int64_t k, int64_t rows, size_t smpbo) {
    const int64_t rows_padded = GGML_PAD(rows, 4);
    const int64_t scale_rows  = (rows * (int64_t) sizeof(float) + k - 1) / k;
    return q->op == GGML_OP_QUANTIZE_I8_CONVROT && q->type == GGML_TYPE_I8 && ggml_is_contiguous(q) &&
           ggml_get_op_params_i32(q, 0) == 256 && k % 256 == 0 && q->ne[0] == k && q->ne[1] == rows_padded + scale_rows &&
           (size_t) k * sizeof(float) + 128 <= smpbo;
}

// The int8 rows of a QUANTIZE_I8_CONVROT output and where its scales go; the padding rows are zeroed.
static std::pair<int8_t *, float *> ggml_cuda_i8_convrot_output(ggml_tensor * q, int64_t k, int64_t rows, cudaStream_t stream) {
    int8_t *      data        = (int8_t *) q->data;
    const int64_t rows_padded = GGML_PAD(rows, 4);
    if (rows_padded > rows) {
        CUDA_CHECK(cudaMemsetAsync(data + k * rows, 0, (size_t) k * (rows_padded - rows), stream));
    }
    return { data, (float *) (data + k * rows_padded) };
}

// The packed int8 input of an own fused GEMM (gemm-i8.cuh; act scales after the rows padded to 4). The GEMM reads it
// until its last block, and the allocator may have handed its buffer on to one of the fused outputs: then it reads a copy.
static const int8_t * ggml_cuda_gemm_i8_input(const ggml_tensor * x, std::initializer_list<const ggml_tensor *> outputs,
                                              ggml_cuda_pool_alloc<int8_t> & copy, cudaStream_t stream) {
    const char * begin = (const char *) x->data;
    const size_t bytes = ggml_nbytes(x);
    for (const ggml_tensor * t : outputs) {
        const char * out = (const char *) t->data;
        if (out < begin + bytes && begin < out + ggml_nbytes(t)) {
            CUDA_CHECK(cudaMemcpyAsync(copy.alloc(bytes), begin, bytes, cudaMemcpyDeviceToDevice, stream));
            return copy.get();
        }
    }
    return (const int8_t *) begin;
}

int ggml_cuda_try_fuse_dit_i8(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node   = cgraph->nodes[i];
    const size_t        smpbo  = ggml_cuda_info().devices[ctx.device].smpbo;
    cudaStream_t        stream = ctx.stream();
    const auto f32_vector = [](const ggml_tensor * t, int64_t n) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == n;
    };

    // RMS_NORM -> MUL(w) -> ADD(b) -> QUANTIZE_I8_CONVROT: a modulated norm feeding an int8 linear.
    if (node->op == GGML_OP_RMS_NORM &&
            ggml_cuda_can_fuse_chain(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_QUANTIZE_I8_CONVROT })) {
        const ggml_tensor * x   = node->src[0];
        const ggml_tensor * mul = cgraph->nodes[i + 1];
        const ggml_tensor * add = cgraph->nodes[i + 2];
        ggml_tensor *       q   = cgraph->nodes[i + 3];
        const int64_t k    = x->ne[0];
        const int64_t rows = ggml_nrows(x);
        if (x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && mul->src[0] == node && add->src[0] == mul &&
                f32_vector(mul->src[1], k) && f32_vector(add->src[1], k) && ggml_cuda_i8_convrot_output_ok(q, k, rows, smpbo)) {
            float eps;
            memcpy(&eps, node->op_params, sizeof(float));
            const auto [q_data, q_scales] = ggml_cuda_i8_convrot_output(q, k, rows, stream);
            ggml_cuda_rms_norm_mul_add_quantize_i8_convrot((const float *) x->data, (const float *) mul->src[1]->data,
                                                           (const float *) add->src[1]->data, eps, k, rows, q_data, q_scales, stream);
            return 3;
        }
    }

    // The rest starts at an int8 linear whose input is already quantized.
    if (node->op != GGML_OP_MUL_MAT || node->src[0]->type != GGML_TYPE_I8 || node->src[1]->type != GGML_TYPE_I8 ||
            node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node) || node->ne[2] != 1 || node->ne[3] != 1 ||
            !ggml_cuda_i8_blas_available(ggml_cuda_info().devices[ctx.device].cc)) {
        return 0;
    }
    const float * weight_scales = node->src[2] != nullptr ? (const float *) node->src[2]->data : nullptr;
    const float * bias          = node->src[3] != nullptr ? (const float *) node->src[3]->data : nullptr;
    const int64_t rows          = ggml_nrows(node);

    // MUL_MAT(int8 qkv) feeding attention: q = VIEW -> RMS_NORM -> MUL -> ROPE_APPLY, k = the same -> CPY(f16) and
    // v = VIEW -> PERMUTE -> CPY(f16), in this order. One kernel turns the int32 sums into the three attention inputs.
    if (i + 12 < cgraph->n_nodes) {
        ggml_tensor * const * nd = cgraph->nodes + i;
        const ggml_op expected[13] = { GGML_OP_MUL_MAT, GGML_OP_VIEW, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE_APPLY,
                                       GGML_OP_VIEW, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE_APPLY, GGML_OP_CPY,
                                       GGML_OP_VIEW, GGML_OP_PERMUTE, GGML_OP_CPY };
        bool match = true;
        for (int j = 0; j < 13 && match; ++j) {
            match = nd[j]->op == expected[j] && (nd[j]->flags & GGML_TENSOR_FLAG_COMPUTE);
        }
        const ggml_tensor * view[3] = { nd[1], nd[5], nd[10] };
        const ggml_tensor * norm[2] = { nd[2], nd[6] };
        const ggml_tensor * mul[2]  = { nd[3], nd[7] };
        const ggml_tensor * rope[2] = { nd[4], nd[8] };
        const int64_t head_dim = view[0]->ne[0];
        const int64_t heads    = view[0]->ne[1];
        const int64_t dim      = head_dim * heads;
        match = match && node->ne[0] == 3 * dim && ggml_node_get_use_count(cgraph, i) == 3 && head_dim % 2 == 0;
        for (int j = 0; j < 3 && match; ++j) {
            match = view[j]->src[0] == node && view[j]->view_offs == (size_t) (j * dim) * sizeof(float) &&
                    view[j]->ne[0] == head_dim && view[j]->ne[1] == heads && view[j]->ne[2] == rows && view[j]->ne[3] == 1 &&
                    view[j]->nb[0] == sizeof(float) && view[j]->nb[1] == head_dim * sizeof(float) && view[j]->nb[2] == node->nb[1] &&
                    ggml_node_get_use_count(cgraph, i + (j == 0 ? 1 : j == 1 ? 5 : 10)) == 1;
        }
        for (int j = 0; j < 2 && match; ++j) {
            const int at = j == 0 ? 2 : 6;
            match = norm[j]->src[0] == view[j] && mul[j]->src[0] == norm[j] && rope[j]->src[0] == mul[j] &&
                    rope[j]->src[1] == rope[0]->src[1] && ggml_get_op_params_i32(rope[j], 0) == ggml_get_op_params_i32(rope[0], 0) &&
                    rope[j]->type == GGML_TYPE_F32 && ggml_is_contiguous(rope[j]) && f32_vector(mul[j]->src[1], head_dim) &&
                    ggml_node_get_use_count(cgraph, i + at) == 1 && ggml_node_get_use_count(cgraph, i + at + 1) == 1;
        }
        const ggml_tensor * pe = rope[0]->src[1];
        match = match && pe->type == GGML_TYPE_F32 && pe->ne[0] == 2 && pe->ne[1] == 2 && pe->ne[2] == head_dim / 2 &&
                pe->ne[3] == rows && nd[9]->src[0] == rope[1] && nd[9]->type == GGML_TYPE_F16 && ggml_is_contiguous(nd[9]) &&
                ggml_node_get_use_count(cgraph, i + 8) == 1 && nd[11]->src[0] == view[2] && nd[12]->src[0] == nd[11] &&
                nd[11]->ne[0] == head_dim && nd[11]->ne[1] == rows && nd[11]->ne[2] == heads &&
                nd[12]->type == GGML_TYPE_F16 && ggml_is_contiguous(nd[12]) && ggml_are_same_shape(nd[12], nd[11]) &&
                ggml_node_get_use_count(cgraph, i + 11) == 1;
        if (match) {
            float q_eps;
            float k_eps;
            memcpy(&q_eps, norm[0]->op_params, sizeof(float));
            memcpy(&k_eps, norm[1]->op_params, sizeof(float));
            // One fused GEMM, no int32 sums in memory (the allocator often hands its input's buffer to k).
            const int64_t k = node->src[1]->ne[0];
            if (head_dim == 128 && ggml_cuda_gemm_i8_supported(ggml_cuda_info().devices[ctx.device].cc, dim, k) &&
                    ggml_mul_mat_has_packed_i8_input(node)) {
                ggml_cuda_pool_alloc<int8_t> x_copy(ctx.pool());
                const int8_t * xq = ggml_cuda_gemm_i8_input(node->src[1], { rope[0], nd[9], nd[12] }, x_copy, stream);
                const float *  xs = (const float *) (xq + k * GGML_PAD(rows, 4));
                ggml_cuda_gemm_i8_qkv_attn_prologue((const int8_t *) node->src[0]->data, xq, xs, weight_scales, bias, rows, k,
                                                    (const float *) mul[0]->src[1]->data, (const float *) mul[1]->src[1]->data,
                                                    q_eps, k_eps, (const float *) pe->data, pe->nb[0] / sizeof(float),
                                                    pe->nb[1] / sizeof(float), pe->nb[2] / sizeof(float), pe->nb[3] / sizeof(float),
                                                    ggml_get_op_params_i32(rope[0], 0) != 0, heads,
                                                    (float *) rope[0]->data, (half *) nd[9]->data, (half *) nd[12]->data, stream);
                return 12;
            }
            ggml_cuda_i8_sums sums(ctx.pool());
            ggml_cuda_mul_mat_i8_sums(ctx, node->src[0], node->src[1], node, sums);
            ggml_cuda_qkv_i32_attn_prologue(sums.sums.get(), sums.act_scales, weight_scales, bias,
                                            (const float *) mul[0]->src[1]->data, (const float *) mul[1]->src[1]->data,
                                            q_eps, k_eps, (const float *) pe->data, pe->nb[0] / sizeof(float),
                                            pe->nb[1] / sizeof(float), pe->nb[2] / sizeof(float), pe->nb[3] / sizeof(float),
                                            ggml_get_op_params_i32(rope[0], 0) != 0, head_dim, heads, rows,
                                            (float *) rope[0]->data, (half *) nd[9]->data, (half *) nd[12]->data, stream);
            return 12;
        }
    }

    // MUL_MAT(int8) -> GLU(situ) [-> QUANTIZE_I8_CONVROT]: the FFN gate, reading the int32 sums.
    if (ggml_cuda_can_fuse_chain(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_GLU })) {
        ggml_tensor * glu = cgraph->nodes[i + 1];
        const int64_t nc  = glu->ne[0];
        if (ggml_get_glu_op(glu) == GGML_GLU_OP_SITU && glu->src[0] == node && glu->src[1] == nullptr &&
                glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu) && node->ne[0] == 2 * nc && nc % 4 == 0) {
            const bool  swapped  = ggml_get_op_params_i32(glu, 1) != 0;
            const float gate_cap = ggml_get_op_params_f32(glu, 2);
            const float up_cap   = ggml_get_op_params_f32(glu, 3);
            ggml_tensor * q      = i + 2 < cgraph->n_nodes ? cgraph->nodes[i + 2] : nullptr;
            const bool quantize  = q != nullptr && q->src[0] == glu &&
                                   ggml_cuda_can_fuse_chain(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_QUANTIZE_I8_CONVROT }) &&
                                   ggml_cuda_i8_convrot_output_ok(q, nc, rows, smpbo);
            ggml_cuda_i8_sums sums(ctx.pool());
            ggml_cuda_mul_mat_i8_sums(ctx, node->src[0], node->src[1], node, sums);
            if (quantize) {
                const auto [q_data, q_scales] = ggml_cuda_i8_convrot_output(q, nc, rows, stream);
                ggml_cuda_situ_i32_quantize_i8_convrot(sums.sums.get(), sums.act_scales, weight_scales, bias, nc, rows,
                                                       swapped, gate_cap, up_cap, q_data, q_scales, stream);
                return 2;
            }
            ggml_cuda_situ_i32(sums.sums.get(), sums.act_scales, weight_scales, bias, nc, rows, swapped, gate_cap, up_cap,
                               (float *) glu->data, stream);
            return 1;
        }
    }

    // MUL_MAT(int8) -> MUL(gate) -> ADD(x): a gated residual.
    if (ggml_cuda_can_fuse_chain(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL, GGML_OP_ADD })) {
        const ggml_tensor * mul = cgraph->nodes[i + 1];
        ggml_tensor *       add = cgraph->nodes[i + 2];
        const ggml_tensor * x   = add->src[0] == mul ? add->src[1] : add->src[0];
        const int64_t n = node->ne[0];
        if (mul->src[0] == node && f32_vector(mul->src[1], n) && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) &&
                ggml_are_same_shape(x, node) && add->type == GGML_TYPE_F32 && ggml_is_contiguous(add) && n % 4 == 0) {
            // -> RMS_NORM -> MUL(w) -> ADD(b) -> QUANTIZE_I8_CONVROT: the next int8 linear's modulated norm of the new
            // residual, quantized right after it.
            const bool chain = ggml_cuda_can_fuse_chain(cgraph, i + 3, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_QUANTIZE_I8_CONVROT });
            const ggml_tensor * norm     = chain ? cgraph->nodes[i + 3] : nullptr;
            const ggml_tensor * norm_mul = chain ? cgraph->nodes[i + 4] : nullptr;
            const ggml_tensor * norm_add = chain ? cgraph->nodes[i + 5] : nullptr;
            ggml_tensor *       q        = chain ? cgraph->nodes[i + 6] : nullptr;
            const bool quantize = chain && norm->src[0] == add && norm_mul->src[0] == norm && norm_add->src[0] == norm_mul &&
                                  f32_vector(norm_mul->src[1], n) && f32_vector(norm_add->src[1], n) &&
                                  ggml_cuda_i8_convrot_output_ok(q, n, rows, smpbo);
            float eps = 0.0f;
            if (quantize) {
                memcpy(&eps, norm->op_params, sizeof(float));
            }
            const auto apart = [](const ggml_tensor * a, const ggml_tensor * b) {
                const char * pa = (const char *) a->data;
                const char * pb = (const char *) b->data;
                return pa >= pb + ggml_nbytes(b) || pb >= pa + ggml_nbytes(a);
            };
            const auto aligned = [](const ggml_tensor * t) { return (uintptr_t) t->data % 16 == 0; };
            const int64_t k = node->src[1]->ne[0];

            // The own GEMM writes the new residual from its accumulators (no int32 sums in memory) and the norm kernel
            // quantizes it from there, so the int8 output must not overlap it.
            if (ggml_cuda_gemm_i8_supported(ggml_cuda_info().devices[ctx.device].cc, n, k) &&
                    ggml_mul_mat_has_packed_i8_input(node) && aligned(x) && aligned(add) && (!quantize || apart(q, add))) {
                ggml_cuda_pool_alloc<int8_t> x_copy(ctx.pool());
                const int8_t * xq = ggml_cuda_gemm_i8_input(node->src[1], { add }, x_copy, stream);
                ggml_cuda_gemm_i8_mul_add((const int8_t *) node->src[0]->data, xq, (const float *) (xq + k * GGML_PAD(rows, 4)),
                                          weight_scales, bias, rows, k, n, (const float *) mul->src[1]->data,
                                          (const float *) x->data, (float *) add->data, stream);
                if (!quantize) {
                    return 2;
                }
                const auto [q_data, q_scales] = ggml_cuda_i8_convrot_output(q, n, rows, stream);
                ggml_cuda_rms_norm_mul_add_quantize_i8_convrot((const float *) add->data, (const float *) norm_mul->src[1]->data,
                                                               (const float *) norm_add->src[1]->data, eps, n, rows, q_data,
                                                               q_scales, stream);
                return 6;
            }

            ggml_cuda_i8_sums sums(ctx.pool());
            ggml_cuda_mul_mat_i8_sums(ctx, node->src[0], node->src[1], node, sums);
            // The residual and the quantization in one kernel, unless the int8 output reuses the memory of x, still being read.
            if (quantize && apart(q, x)) {
                const auto [q_data, q_scales] = ggml_cuda_i8_convrot_output(q, n, rows, stream);
                if (ggml_cuda_mul_add_i32_rms_norm_quantize_i8_convrot(
                        sums.sums.get(), sums.act_scales, weight_scales, bias, (const float *) mul->src[1]->data,
                        (const float *) x->data, (float *) add->data, n, rows, (const float *) norm_mul->src[1]->data,
                        (const float *) norm_add->src[1]->data, eps, q_data, q_scales, stream)) {
                    return 6;
                }
            }
            ggml_cuda_mul_add_i32(sums.sums.get(), sums.act_scales, weight_scales, bias, (const float *) mul->src[1]->data,
                                  (const float *) x->data, (float *) add->data, n, rows, stream);
            return 2;
        }
    }
    return 0;
}
#endif // !defined(GGML_USE_MUSA)
