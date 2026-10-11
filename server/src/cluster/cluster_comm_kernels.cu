// cluster_comm_kernels.cu - device kernels for the RCCL cluster communicator.
//
// Compiled as HIP (LANGUAGE HIP in server/CMakeLists.txt) only when
// LUCE_CLUSTER=ON. Provides the f32 <-> bf16 conversions used by
// allreduce_sum_bf16_compressed (bf16 is written by hand, round-to-nearest-
// even on the raw bits, so this file needs neither hip_bf16.h nor ggml) and
// the 8-bit blocks of allreduce_sum_q8_exchange.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace luce::cluster {

namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        // Inf/NaN: truncate, keep NaN quiet.
        uint16_t r = (uint16_t)(u >> 16);
        if (u & 0x007fffffu) r |= 0x0040u;
        return r;
    }
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

__global__ void k_f32_to_bf16(const float * __restrict__ in, uint16_t * __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i < n) out[i] = f32_to_bf16_bits(in[i]);
}

__global__ void k_bf16_to_f32(const uint16_t * __restrict__ in, float * __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t)in[i] << 16);
}

// One thread per block of 32 values: the scale is the block's absolute
// maximum over 127, the values round to nearest.
__global__ void k_f32_to_q8(const float * __restrict__ in, int8_t * __restrict__ q,
                            float * __restrict__ d, size_t nb) {
    const size_t b = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (b >= nb) return;
    const float * x = in + b * 32;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[i]));
    const float scale = amax / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    int8_t * qb = q + b * 32;
    for (int i = 0; i < 32; ++i) qb[i] = (int8_t)__float2int_rn(x[i] * inv);
    d[b] = scale;
}

// out[i] = the ranks' dequantized values summed in rank order.
__global__ void k_q8_sum(const uint8_t * __restrict__ slots, int ranks, size_t slot_bytes,
                         size_t n, float * __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (int r = 0; r < ranks; ++r) {
        const uint8_t * slot = slots + (size_t)r * slot_bytes;
        const float scale = ((const float *)(slot + n))[i / 32];
        acc += (float)((const int8_t *)slot)[i] * scale;
    }
    out[i] = acc;
}

unsigned grid_for(size_t n) {
    const size_t blocks = (n + (size_t)kBlock - 1) / (size_t)kBlock;
    return blocks > 0x7fffffffull ? 0x7fffffffu : (unsigned)blocks;
}

}  // namespace

// Declared in cluster_comm.cpp under LUCE_CLUSTER_RCCL.
bool cluster_kernel_f32_to_bf16(const float * in, uint16_t * out, size_t n, void * stream) {
    if (n == 0) return true;
    k_f32_to_bf16<<<grid_for(n), kBlock, 0, (hipStream_t)stream>>>(in, out, n);
    return hipGetLastError() == hipSuccess;
}

bool cluster_kernel_bf16_to_f32(const uint16_t * in, float * out, size_t n, void * stream) {
    if (n == 0) return true;
    k_bf16_to_f32<<<grid_for(n), kBlock, 0, (hipStream_t)stream>>>(in, out, n);
    return hipGetLastError() == hipSuccess;
}

// `slot`: n int8 values, then n / 32 f32 scales (q8_exchange_slot_bytes).
bool cluster_kernel_f32_to_q8(const float * in, void * slot, size_t n, void * stream) {
    if (n == 0) return true;
    const size_t nb = n / 32;
    k_f32_to_q8<<<grid_for(nb), kBlock, 0, (hipStream_t)stream>>>(
        in, (int8_t *)slot, (float *)((uint8_t *)slot + n), nb);
    return hipGetLastError() == hipSuccess;
}

bool cluster_kernel_q8_sum(const void * slots, int ranks, size_t slot_bytes, size_t n, float * out,
                           void * stream) {
    if (n == 0) return true;
    k_q8_sum<<<grid_for(n), kBlock, 0, (hipStream_t)stream>>>(
        (const uint8_t *)slots, ranks, slot_bytes, n, out);
    return hipGetLastError() == hipSuccess;
}

}  // namespace luce::cluster
