#pragma once
#include "common.cuh"

// Stride-1 2D convolution with f16 weights on RDNA4 WMMA (the convolutions of the image VAEs). The bias and a residual,
// when given, are added in the epilogue; out may be the conv's own data or a fused consumer's.
bool ggml_cuda_conv2d_rdna4_supported(int cc, const ggml_tensor * conv);
void ggml_cuda_conv2d_rdna4(ggml_backend_cuda_context & ctx, const ggml_tensor * conv, const float * bias, float * out);

// x_nhwc: the input as f16 channels-last, already computed by a fused producer (null: convert the conv's f32 src[1]).
// up2x: x_nhwc holds the source of a nearest 2x upscale that the conv's src[1] is, read at (y / 2, x / 2).
// res: [N][OC][OH * OW] f32 added after the bias, or null.
void ggml_cuda_conv2d_rdna4_ex(ggml_backend_cuda_context & ctx, const ggml_tensor * conv, const half * x_nhwc, bool up2x,
                               const float * bias, const float * res, float * out);

// [N][C][HW] f32 -> [N][HW][C] f16, the conversion of the conv's input pass.
void ggml_cuda_conv2d_rdna4_to_nhwc(cudaStream_t stream, const float * x, half * y, int C, int HW, int N);

// CONV_2D at node i followed by its bias ADD and optionally a residual ADD of the result and an earlier tensor (resnet
// and attention skips). Returns the index of the last node taken, or -1 when there is no bias ADD to take.
int ggml_cuda_conv2d_rdna4_match(const ggml_cgraph * cgraph, int i, const float ** bias, const float ** res, float ** out);

// Nearest 2x UPSCALE whose only consumer is an RDNA4 conv (+ bias): the conv reads the source pixels directly.
// Returns the number of graph nodes consumed after node i, 0 when the nodes from i do not match.
int ggml_cuda_try_fuse_upscale_conv(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
