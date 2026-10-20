#pragma once

#include "common.cuh"

// Convrot int8 quantization with the row in the registers of one wave (or two, convrot_wave_row), instead of a
// 1024-thread block with the row in shared memory, so many rows are in flight. The 256-point transform runs the same
// radix-4 butterflies in the same order and with the same float operations as ggml_cuda_convrot_quantize_row_i8, so
// the int8 values and scales are identical.
//
// A row of K = 512 P values is P pairs of 256-groups. For pair p, lane l holds group 2 p + l / 16, elements
// 64 q + 4 (l % 16) + i (q, i = 0..3): in base 4 the element's digits are (i, l % 4, l / 4 % 4, q), so the stride-1 and
// stride-64 butterflies are in-lane and the stride-4 and stride-16 ones exchange with lanes l ^ 1..3 and l ^ 4, 8, 12
// (DPP row_xmask). Every 16-byte load and 4-byte store of a half-wave is contiguous.

template <int ctrl>
static __device__ __forceinline__ float convrot_wave_dpp(const float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), ctrl, 0xf, 0xf, false));
}

// In-lane radix-4 butterfly: a, b, c, d are the elements whose digit is 0, 1, 2, 3.
static __device__ __forceinline__ void convrot_wave_r4(float & a, float & b, float & c, float & d) {
    const float a0 = a, b0 = b, c0 = c, d0 = d;
    a =  a0 + b0 + c0 - d0;
    b =  a0 + b0 - c0 + d0;
    c =  a0 - b0 + c0 + d0;
    d = -a0 + b0 + c0 + d0;
}

// The same butterfly across the 4 lanes l ^ (m * s), m = 0..3; r is this lane's digit. The lane with digit p holds
// v_p = x_{p ^ r}, and output r takes the minus sign on v_{3 - r}: ((v0 + v1) + v2) - v3 for r = 0, and so on.
template <int s>
static __device__ __forceinline__ float convrot_wave_r4_lanes(const float x0, const int r) {
    const float x1 = convrot_wave_dpp<0x160 | (1 * s)>(x0);
    const float x2 = convrot_wave_dpp<0x160 | (2 * s)>(x0);
    const float x3 = convrot_wave_dpp<0x160 | (3 * s)>(x0);
    const bool  r1 = r & 1;
    const bool  r2 = r & 2;
    const float v0 = r2 ? (r1 ? x3 : x2) : (r1 ? x1 : x0);
    const float v1 = r2 ? (r1 ? x2 : x3) : (r1 ? x0 : x1);
    const float v2 = r2 ? (r1 ? x1 : x0) : (r1 ? x3 : x2);
    const float v3 = r2 ? (r1 ? x0 : x1) : (r1 ? x2 : x3);
    return (((r == 3 ? -v0 : v0) + (r == 2 ? -v1 : v1)) + (r == 1 ? -v2 : v2)) + (r == 0 ? -v3 : v3);
}

static __device__ __forceinline__ int64_t convrot_wave_index(const int p, const int q) {
    const int lane = threadIdx.x % 32;
    return (int64_t) (2 * p + lane / 16) * 256 + 64 * q + 4 * (lane % 16);
}

// The 256-point transform of the two groups of one pair, held as v[q][i], in place.
static __device__ __forceinline__ void convrot_wave_rotate_pair(float (&v)[4][4]) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        convrot_wave_r4(v[q][0], v[q][1], v[q][2], v[q][3]);
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[q][i] = convrot_wave_r4_lanes<1>(v[q][i], lane % 4);
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[q][i] = convrot_wave_r4_lanes<4>(v[q][i], lane / 4 % 4);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        convrot_wave_r4(v[0][i], v[1][i], v[2][i], v[3][i]);
    }
}

// The sum of squares in exactly the order of the 1024-thread block kernel (rms_norm_mul_add_quantize_i8_convrot): its
// thread t = e % 1024 adds elements t, t + 1024, ..., then the 32-lane butterflies over t % 32 and t / 32. Thread t's
// elements are this lane's elements of the pairs p0, p0 + 2, ... (one pass, p0 = 0, 1), t % 32 = i + 4 (l % 8) and
// t / 32 = l / 8 % 2 + 2 q + 8 (l / 16) + 16 p0. A pass of J pairs, pair(j) its j-th, folded over t % 32 into s[q]:
template <int J, typename Pair>
static __device__ __forceinline__ void convrot_wave_sumsq_pass(const Pair & pair, float (&s)[4]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        float t[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            t[i] = 0.0f;
#pragma unroll
            for (int j = 0; j < J; ++j) {
                t[i] = __builtin_fmaf(pair(j)[q][i], pair(j)[q][i], t[i]);  // the block kernel's loop is contracted
            }
            t[i] += convrot_wave_dpp<0x160 | 4>(t[i]);  // 16, 8, 4 are lanes ^ 4, 2, 1
            t[i] += convrot_wave_dpp<0x160 | 2>(t[i]);
            t[i] += convrot_wave_dpp<0x160 | 1>(t[i]);
        }
        s[q] = (t[0] + t[2]) + (t[1] + t[3]);  // 2, 1 are i ^ 2, i ^ 1
    }
}

// ... and the two passes over t / 32: 16 is p0, 8 is lane ^ 16, 4 and 2 are q ^ 2, q ^ 1, 1 is lane ^ 8.
static __device__ __forceinline__ float convrot_wave_sumsq_combine(const float (&s0)[4], const float (&s1)[4]) {
    float z[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        z[q] = s0[q] + s1[q];
        z[q] += __shfl_xor(z[q], 16, 32);
    }
    const float z0 = z[0] + z[2];
    const float z1 = z[1] + z[3];
    float sumsq = z0 + z1;
    sumsq += convrot_wave_dpp<0x160 | 8>(sumsq);
    return sumsq;
}

// A row of K = 512 P values -> rms_norm(row) * w + b if NORM, then the convrot int8 values at dst_row and the scale
// (absolute max / 127). H waves share the row: H = 1, or 2 with one row per 64-thread block, where wave h holds the
// pairs h, h + 2, ..., which is exactly one sum-of-squares pass, so the passes meet in shared memory. load(p, q)
// returns this lane's 4 values of sub-block q of pair p.
template <int P, int H, bool NORM, typename Load>
static __device__ __forceinline__ void convrot_wave_row(
        const Load & load, const float * w, const float * b, const float eps, int8_t * dst_row, float * scale_out) {
    static_assert(H == 1 || H == 2, "one or two waves per row");
    constexpr int K    = 512 * P;
    constexpr int J    = P / H;  // pairs per wave
    const int     lane = threadIdx.x % 32;
    const int     h    = H == 1 ? 0 : threadIdx.x / 32;
    const auto    pair_of = [&](const int j) { return H * j + h; };
    float v[J][4][4];
#pragma unroll
    for (int j = 0; j < J; ++j) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 t = load(pair_of(j), q);
            v[j][q][0] = t.x; v[j][q][1] = t.y; v[j][q][2] = t.z; v[j][q][3] = t.w;
        }
    }
    if constexpr (NORM) {
        float s0[4];
        float s1[4];
        if constexpr (H == 1) {
            convrot_wave_sumsq_pass<P / 2>([&](const int j) -> const float (&)[4][4] { return v[2 * j]; }, s0);
            convrot_wave_sumsq_pass<P / 2>([&](const int j) -> const float (&)[4][4] { return v[2 * j + 1]; }, s1);
        } else {
            __shared__ float partial[2][4][32];
            float s[4];
            convrot_wave_sumsq_pass<J>([&](const int j) -> const float (&)[4][4] { return v[j]; }, s);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                partial[h][q][lane] = s[q];
            }
            __syncthreads();
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                s0[q] = partial[0][q][lane];
                s1[q] = partial[1][q][lane];
            }
        }
        const float scale = rsqrtf(convrot_wave_sumsq_combine(s0, s1) / K + eps);
#pragma unroll
        for (int j = 0; j < J; ++j) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 wq = *(const float4 *) (w + convrot_wave_index(pair_of(j), q));
                const float4 bq = *(const float4 *) (b + convrot_wave_index(pair_of(j), q));
                v[j][q][0] = (scale * v[j][q][0] * wq.x + bq.x) * (1.0f / 16.0f);
                v[j][q][1] = (scale * v[j][q][1] * wq.y + bq.y) * (1.0f / 16.0f);
                v[j][q][2] = (scale * v[j][q][2] * wq.z + bq.z) * (1.0f / 16.0f);
                v[j][q][3] = (scale * v[j][q][3] * wq.w + bq.w) * (1.0f / 16.0f);
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < J; ++j) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    v[j][q][i] *= 1.0f / 16.0f;
                }
            }
        }
    }

    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < J; ++j) {
        convrot_wave_rotate_pair(v[j]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                amax = fmaxf(amax, fabsf(v[j][q][i]));
            }
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    if constexpr (H == 2) {
        __shared__ float maxima[2];
        if (lane == 0) {
            maxima[h] = amax;
        }
        __syncthreads();
        amax = fmaxf(maxima[0], maxima[1]);
    }
    const float row_scale = amax / 127.0f;
    if (lane == 0 && h == 0) {
        *scale_out = row_scale;
    }
    const float inv_scale = row_scale == 0.0f ? 0.0f : 1.0f / row_scale;
    const auto  quantize  = [&](const float x) {
        const int q = row_scale == 0.0f ? 0 : __float2int_rn(x * inv_scale);
        return (uint32_t) (uint8_t) (int8_t) max(-127, min(127, q));
    };
#pragma unroll
    for (int j = 0; j < J; ++j) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            *(uint32_t *) (dst_row + convrot_wave_index(pair_of(j), q)) =
                quantize(v[j][q][0]) | (quantize(v[j][q][1]) << 8) | (quantize(v[j][q][2]) << 16) | (quantize(v[j][q][3]) << 24);
        }
    }
}

// Rows of K = 512 P f32 values -> convrot int8 rows (stride dst_stride) and scales; with NORM first rms_norm * w + b.
template <int P, bool NORM>
static __global__ void __launch_bounds__(256) convrot_wave_kernel(
        const float * x, const float * w, const float * b, const float eps, const int64_t rows, int8_t * dst,
        const int64_t dst_stride, float * scales) {
#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4))
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) {
        return;
    }
    const float * x_row = x + row * (512 * P);
    convrot_wave_row<P, 1, NORM>([&](const int p, const int q) { return *(const float4 *) (x_row + convrot_wave_index(p, q)); },
                                 w, b, eps, dst + row * dst_stride, scales + row);
#else
    GGML_UNUSED_VARS(x, w, b, eps, rows, dst, dst_stride, scales);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4))
}

// Rows of k f32 values -> convrot int8 rows (stride dst_stride) and their scales, rms_norm(x) * w + b first when w is
// given. Returns false (nothing launched) unless the device is RDNA2+ and k is 2048, 3072 or 4096.
static bool ggml_cuda_convrot_wave(const float * x, const float * w, const float * b, const float eps, const int64_t k,
                                   const int64_t rows, int8_t * dst, const int64_t dst_stride, float * scales,
                                   cudaStream_t stream) {
    const int  cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const auto aligned = [](const void * p) { return p == nullptr || (uintptr_t) p % 16 == 0; };
    if (!(GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) || scales == nullptr ||
            (w == nullptr) != (b == nullptr) || !aligned(x) || !aligned(w) || !aligned(b) || dst_stride % 4 != 0 ||
            (uintptr_t) dst % 4 != 0) {
        return false;
    }
    const dim3 grid((rows + 7) / 8);
    const auto launch = [&](auto kernel) {
        kernel<<<grid, 256, 0, stream>>>(x, w, b, eps, rows, dst, dst_stride, scales);
        return true;
    };
    switch (k) {
        case 2048: return w ? launch(convrot_wave_kernel<4, true>) : launch(convrot_wave_kernel<4, false>);
        case 3072: return w ? launch(convrot_wave_kernel<6, true>) : launch(convrot_wave_kernel<6, false>);
        case 4096: return w ? launch(convrot_wave_kernel<8, true>) : launch(convrot_wave_kernel<8, false>);
        default:   return false;
    }
}
