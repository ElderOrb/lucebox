#pragma once

#include "common.cuh"

#include <initializer_list>

// Kernels that fuse int8 (convrot) linears with their neighbours in a DiT block: the norm before a linear writes its
// int8 input directly, and the op after it reads the int32 GEMM sums, so neither the f32 activation nor the f32 GEMM
// output makes an extra round trip through memory. The sums are dequantized as (float) sum * activation_scale[row]
// * weight_scale[col] + bias[col]; weight_scales and bias may be null.

// rms_norm(x) * w + b on rows of k values, quantized with convrot (k % 256 == 0) into dst (rows of k int8) and scales.
void ggml_cuda_rms_norm_mul_add_quantize_i8_convrot(
    const float * x, const float * w, const float * b, float eps, int64_t k, int64_t rows,
    int8_t * dst, float * scales, cudaStream_t stream);

// The situ gate on int32 sums laid out [2*nc, rows]; the gate is the second half when swapped. Writes f32 [nc, rows].
void ggml_cuda_situ_i32(
    const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
    int64_t nc, int64_t rows, bool swapped, float gate_cap, float up_cap, float * dst, cudaStream_t stream);

// The same, quantized with convrot for the next int8 GEMM (nc % 256 == 0): rows of nc int8 values and their scales.
void ggml_cuda_situ_i32_quantize_i8_convrot(
    const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
    int64_t nc, int64_t rows, bool swapped, float gate_cap, float up_cap, int8_t * dst, float * scales,
    cudaStream_t stream);

// The attention inputs from the int32 sums of a fused qkv linear, laid out [3 * heads * head_dim, rows]: q and k go
// through rms_norm * w and rope_apply (pe [2, 2, head_dim / 2, rows], element strides pe_s0..3), v is only cast.
// Writes q (f32), k and v (f16), each [head_dim, rows, heads] contiguous.
void ggml_cuda_qkv_i32_attn_prologue(
    const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
    const float * q_w, const float * k_w, float q_eps, float k_eps, const float * pe,
    int64_t pe_s0, int64_t pe_s1, int64_t pe_s2, int64_t pe_s3, bool interleaved,
    int64_t head_dim, int64_t heads, int64_t rows, float * q, half * k, half * v, cudaStream_t stream);

// x + y * gate with y the dequantized int32 sums [n, rows] and gate per column (n % 4 == 0): a gated residual.
void ggml_cuda_mul_add_i32(
    const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
    const float * gate, const float * x, float * dst, int64_t n, int64_t rows, cudaStream_t stream);

// The same residual written to dst and, from it, rms_norm * w + b quantized with convrot into q_dst (rows of n int8)
// and q_scales, in one kernel. q_dst must not overlap x. Returns false (nothing launched) unless the device is RDNA2+
// and n is 2048, 3072 or 4096.
bool ggml_cuda_mul_add_i32_rms_norm_quantize_i8_convrot(
    const int32_t * sums, const float * act_scales, const float * weight_scales, const float * bias,
    const float * gate, const float * x, float * dst, int64_t n, int64_t rows,
    const float * w, const float * b, float eps, int8_t * q_dst, float * q_scales, cudaStream_t stream);

// The int8 GEMM of a MUL_MAT with I8 weights, before dequantization: int32 sums [n, rows padded to 4] and the
// activation scales. src1 is quantized here unless it already is the output of QUANTIZE_I8_CONVROT. The members are
// declared in allocation order, so the pool buffers are released last in, first out.
struct ggml_cuda_i8_sums {
    ggml_cuda_pool_alloc<int8_t>  qdata;
    ggml_cuda_pool_alloc<float>   scales;
    ggml_cuda_pool_alloc<int32_t> sums;
    const float *                 act_scales = nullptr;

    explicit ggml_cuda_i8_sums(ggml_cuda_pool & pool) : qdata(pool), scales(pool), sums(pool) {}
};

// Defined in ggml-cuda.cu, next to the int8 MUL_MAT it serves.
void ggml_cuda_mul_mat_i8_sums(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
    ggml_cuda_i8_sums & out);

// ggml_can_fuse without its same-shape rule, for chains whose shapes change: ROPE_APPLY writes [head_dim, tokens,
// heads], an int8 MUL_MAT and a GLU change the row length, QUANTIZE_I8_CONVROT packs scales after the rows.
bool ggml_cuda_can_fuse_chain(const ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops);

// The int8 linears of a DiT block fused with their neighbours, from graph node i: the norm before a linear quantizes,
// the GEMM's consumer reads its sums (or runs in its epilogue, gemm-i8.cuh). Returns the number of nodes consumed
// after node i, 0 when nothing matches.
int ggml_cuda_try_fuse_dit_i8(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
