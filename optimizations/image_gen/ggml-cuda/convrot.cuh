#pragma once

#include "common.cuh"

// Convrot int8 quantization of one row held in shared memory and already scaled by 1/16 (the normalization of the
// 256-point Hadamard transform): rotates every 256-value group in place, then writes the row's int8 values to dst_row
// and its scale (absolute max / 127) to *scale_out. Runs on a whole block of 1024 threads; maxima is shared, 32 floats.
// k is a multiple of 256, so the row moves four values per thread with 16-byte LDS reads and 4-byte int8 stores.
static __device__ __forceinline__ void ggml_cuda_convrot_quantize_row_i8(
        float * values, const int64_t k, int8_t * dst_row, float * scale_out, float * maxima) {
    constexpr int group_size      = 256;
    constexpr int group_threads   = group_size / 4;
    constexpr int groups_per_wave = 1024 / group_threads;

    const int tid        = threadIdx.x;
    const int group_lane = tid % group_threads;
    const int wave_group = tid / group_threads;
    const int64_t groups = k / group_size;
    for (int64_t group_base = 0; group_base < groups; group_base += groups_per_wave) {
        const int64_t group = group_base + wave_group;
#pragma unroll
        for (int stride = 1; stride < group_size; stride *= 4) {
            if (group < groups) {
                float * group_values = values + group * group_size;
                const int base       = (group_lane / stride) * 4 * stride + group_lane % stride;
                const int i0         = base;
                const int i1         = i0 + stride;
                const int i2         = i1 + stride;
                const int i3         = i2 + stride;
                const float a        = group_values[i0];
                const float b        = group_values[i1];
                const float c        = group_values[i2];
                const float d        = group_values[i3];
                group_values[i0]     =  a + b + c - d;
                group_values[i1]     =  a + b - c + d;
                group_values[i2]     =  a - b + c + d;
                group_values[i3]     = -a + b + c + d;
            }
            __syncthreads();
        }
    }

    const float4 * values4 = (const float4 *) values;
    float local_max = 0.0f;
    for (int64_t i4 = tid; i4 < k / 4; i4 += blockDim.x) {
        const float4 v = values4[i4];
        local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    local_max = warp_reduce_max<32>(local_max);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        maxima[warp] = local_max;
    }
    __syncthreads();

    if (warp == 0) {
        local_max = lane < blockDim.x / 32 ? maxima[lane] : 0.0f;
        local_max = warp_reduce_max<32>(local_max);
        if (lane == 0) {
            maxima[0]  = local_max / 127.0f;
            *scale_out = maxima[0];
        }
    }
    __syncthreads();

    const float row_scale = maxima[0];
    const float inv_scale = row_scale == 0.0f ? 0.0f : 1.0f / row_scale;
    const auto quantize = [&](const float v) {
        const int q = row_scale == 0.0f ? 0 : __float2int_rn(v * inv_scale);
        return (uint32_t) (uint8_t) (int8_t) max(-127, min(127, q));
    };
    for (int64_t i4 = tid; i4 < k / 4; i4 += blockDim.x) {
        const float4 v = values4[i4];
        ((uint32_t *) dst_row)[i4] = quantize(v.x) | (quantize(v.y) << 8) | (quantize(v.z) << 16) | (quantize(v.w) << 24);
    }
}
