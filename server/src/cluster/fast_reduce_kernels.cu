// fast_reduce_kernels.cu - the device half of the lean decode-path reduction.
//
// The waits are stream memory operations (hipStreamWaitValue32, in
// fast_reduce.cpp); these kernels move and add the rows:
//
//   pack           write this rank's row into pinned memory (bf16, or f32 for
//                  an exact row) and raise its flag.
//   add            add the peer's row straight out of pinned memory.
//   hyb_combine,   the two-rank hybrid exchange (FastReduce::hyb_combine).
//   hyb_final
//
// The flags live in pinned memory, not managed memory: gfx1151 reports XNACK
// disabled, so the GPU cannot take a page fault, and a managed page the CPU
// has touched faults the moment a kernel reads it. Pinned pages never migrate.
//
// A stream wait has no timeout of its own: the host's progress thread bounds
// it (LUCE_CLUSTER_FAST_REDUCE_STALL_S) and releases the GPU when a peer is
// gone.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace luce::cluster {

namespace {
constexpr int kBlock = 256;
constexpr int kMaxGrid = 128;

int grid_for(int n) {
    int grid = (n + kBlock - 1) / kBlock;
    if (grid < 1) grid = 1;
    if (grid > kMaxGrid) grid = kMaxGrid;
    return grid;
}
}  // namespace

// bf16 rows round both operands, so every rank computes the same sum.
__device__ __forceinline__ uint16_t fr_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float fr_bf16_to_f32(uint16_t h) {
    return __uint_as_float(((uint32_t) h) << 16);
}

// The grid that wrote an exchange row publishes its flag itself. Each block
// orders its writes at device scope and counts in; the last makes every
// block's writes visible system-wide with ONE release and stores the flag,
// which is cheaper than a hipStreamWriteValue32 after the kernel. counter is
// this device's, zeroed, and left zeroed for the next grid on the same stream.
__device__ __forceinline__ void fr_publish_last_block(uint32_t * flag, uint32_t value, uint32_t * counter) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t prev = atomicAdd(counter, 1u);
        if (prev == gridDim.x - 1) {
            *counter = 0;
            __threadfence_system();
            __hip_atomic_store(flag, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

__global__ void fast_reduce_pack_kernel(const float * __restrict__ src, void * dst, int n, int bf16,
                                        uint32_t * flag, uint32_t value, uint32_t * counter) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += blockDim.x * gridDim.x) {
        if (bf16) ((uint16_t *) dst)[k] = fr_f32_to_bf16(src[k]);
        else      ((float *) dst)[k] = src[k];
    }
    fr_publish_last_block(flag, value, counter);
}

__global__ void fast_reduce_add_kernel(float * __restrict__ dst, const void * peer, int n, int bf16) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += blockDim.x * gridDim.x) {
        if (bf16) {
            const float own = fr_bf16_to_f32(fr_f32_to_bf16(dst[k]));
            dst[k] = own + fr_bf16_to_f32(((const uint16_t *) peer)[k]);
        } else {
            dst[k] = dst[k] + ((const float *) peer)[k];
        }
    }
}

// Hybrid exchange, iGPU side. own = cold + hot in f32, where hot already holds
// the dGPU's hot experts plus shared expert, then the bf16 wire row; the final
// sum is bf16(own) + peer exactly as fast_reduce_add_kernel forms it.
__global__ void fast_reduce_hyb_combine_kernel(const float * __restrict__ cold, const float * __restrict__ hot,
                                               void * own_row, int n,
                                               uint32_t * flag, uint32_t value, uint32_t * counter) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += blockDim.x * gridDim.x) {
        ((uint16_t *) own_row)[k] = fr_f32_to_bf16(cold[k] + hot[k]);
    }
    fr_publish_last_block(flag, value, counter);
}

__global__ void fast_reduce_hyb_final_kernel(const void * own_row, const void * peer_row, float * __restrict__ dst,
                                             int n, uint32_t * flag, uint32_t value, uint32_t * counter) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += blockDim.x * gridDim.x) {
        dst[k] = fr_bf16_to_f32(((const uint16_t *) own_row)[k]) + fr_bf16_to_f32(((const uint16_t *) peer_row)[k]);
    }
    fr_publish_last_block(flag, value, counter);
}

void fast_reduce_launch_pack(const float * src, void * dst, int n, int bf16,
                             uint32_t * flag, uint32_t value, uint32_t * counter, hipStream_t stream) {
    hipLaunchKernelGGL(fast_reduce_pack_kernel, dim3(grid_for(n)), dim3(kBlock), 0, stream, src, dst, n, bf16,
                       flag, value, counter);
}

void fast_reduce_launch_add(float * dst, const void * peer, int n, int bf16, hipStream_t stream) {
    hipLaunchKernelGGL(fast_reduce_add_kernel, dim3(grid_for(n)), dim3(kBlock), 0, stream, dst, peer, n, bf16);
}

void fast_reduce_launch_hyb_combine(const float * cold, const float * hot, void * own_row, int n,
                                    uint32_t * flag, uint32_t value, uint32_t * counter, hipStream_t stream) {
    hipLaunchKernelGGL(fast_reduce_hyb_combine_kernel, dim3(grid_for(n)), dim3(kBlock), 0, stream,
                       cold, hot, own_row, n, flag, value, counter);
}

void fast_reduce_launch_hyb_final(const void * own_row, const void * peer_row, float * dst, int n,
                                  uint32_t * flag, uint32_t value, uint32_t * counter, hipStream_t stream) {
    hipLaunchKernelGGL(fast_reduce_hyb_final_kernel, dim3(grid_for(n)), dim3(kBlock), 0, stream,
                       own_row, peer_row, dst, n, flag, value, counter);
}

}  // namespace luce::cluster
