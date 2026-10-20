#pragma once

#include "common.cuh"

// Flash attention for long sequences with head size 128 on RDNA4 (fattn-rdna4.cu). Returns false, without launching
// anything, when dst is not a case it handles.
bool ggml_cuda_flash_attn_ext_rdna4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
