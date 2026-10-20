#pragma once
#include "common.cuh"

// GROUP_NORM followed by its per-channel affine MUL and ADD and optionally SILU (GroupNorm32 + swish of the image
// VAEs) as one stats pass and one normalize pass. Returns the number of graph nodes consumed after node i, 0 when the
// nodes from i do not match.
int ggml_cuda_try_fuse_group_norm(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);

// The next node after j that computes something: views (weight and bias reshapes) sit between fusable ops.
int ggml_cuda_next_compute_node(const ggml_cgraph * cgraph, int j);
