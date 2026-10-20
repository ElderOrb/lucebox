#pragma once

#include "common.cuh"

// int8 x int8 -> int32 GEMM through hipBLASLt with a tuned kernel per shape: sums [rows, n] = x [rows, k] * w [n, k]^T,
// n contiguous. hipblasGemmEx's default kernel runs a DiT's qkv shape (9216 x 3072, 4116 tokens) at half the speed of
// hipBLASLt's best. Returns false, without launching anything, when the build has no hipBLASLt or the call fails, and
// the caller runs the GEMM itself.
bool ggml_cuda_gemm_i8_hipblaslt(ggml_backend_cuda_context & ctx, const int8_t * w, const int8_t * x, int32_t * sums,
                                 int64_t n, int64_t rows, int64_t k);
