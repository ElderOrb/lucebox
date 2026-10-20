#pragma once

#include "common.cuh"

// Own int8 GEMM for RDNA4 WMMA with the consumer fused into its epilogue, so the int32 sums never go to memory
// (hipBLASLt on gfx12 can only write them out as i32, ~1 GB per DiT layer written and read back). Same results as the
// hipBLASLt GEMM followed by the dit-i8.cu kernels, bit for bit: int32 sums are exact in any k order and the epilogue
// replays the consumer's float operations in the same order.
//
// All of them read w [n, k] int8 with weight_scales [n] and bias [n] (either may be null) and x [rows, k] int8 with
// act_scales [rows], dequantized as in dit-i8.cuh. Their outputs must not overlap x or act_scales.

// True when the fused GEMMs can run with n output features and k inputs: RDNA4, n % 256 == 0, k % 64 == 0, and not
// disabled with GGML_CUDA_I8_GEMM=0.
bool ggml_cuda_gemm_i8_supported(int cc, int64_t n, int64_t k);

// The qkv linear and the attention prologue (ggml_cuda_qkv_i32_attn_prologue's work on the sums): q and k through
// rms_norm * w and rope_apply (pe [2, 2, 64, rows], element strides pe_s0..3), v cast. w is [3 * heads * 128, k]; q
// (f32), k and v (f16) are [128, rows, heads] contiguous. Needs head_dim 128 and ggml_cuda_gemm_i8_supported with n =
// heads * 128.
void ggml_cuda_gemm_i8_qkv_attn_prologue(
    const int8_t * w, const int8_t * x, const float * act_scales, const float * weight_scales, const float * bias,
    int64_t rows, int64_t k, const float * q_w, const float * k_w, float q_eps, float k_eps, const float * pe,
    int64_t pe_s0, int64_t pe_s1, int64_t pe_s2, int64_t pe_s3, bool interleaved, int64_t heads,
    float * q, half * k_out, half * v, cudaStream_t stream);

// A gated residual, ggml_cuda_mul_add_i32's arithmetic: dst [rows, n] = res + dequantized sums * gate [n]. res and dst
// may be the same tensor.
void ggml_cuda_gemm_i8_mul_add(
    const int8_t * w, const int8_t * x, const float * act_scales, const float * weight_scales, const float * bias,
    int64_t rows, int64_t k, int64_t n, const float * gate, const float * res, float * dst, cudaStream_t stream);
