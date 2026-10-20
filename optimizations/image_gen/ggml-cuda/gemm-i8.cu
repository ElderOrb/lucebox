#include "gemm-i8.cuh"
#include "rope.cuh"

#include <cstdlib>

// Block tile 256 features x 128 tokens x 64 k on 8 waves (4 x 2 waves of 64 x 64, 4 x 4 WMMA 16x16x16 iu8 tiles each).
// Both operands go through double-buffered LDS with one barrier per k block; the next block's global loads are issued
// before the WMMAs and stored to LDS after them. In a 64-byte k block a lane feeds bytes 32 (lane / 16) .. + 31 of its
// row (weights, A) and of its column (tokens, B) as four k steps of 8 bytes: WMMA k 16 s + 8 h + j is real k
// 32 h + 8 s + j for both operands, so every k is summed exactly once and each lane reads 16-byte LDS vectors.
// hipBLASLt-level speed came from three things: the LDS-only fence (a __syncthreads fence invalidates L0 every k block
// in WGP mode), a branch-free k loop (otherwise the WMMAs sink below the next block's LDS stores, exposing the global
// loads), and the 256-wide tile; G token blocks per feature block keep neighbours on the same L2 lines.

typedef int v2i_t __attribute__((ext_vector_type(2)));
typedef int v4i_t __attribute__((ext_vector_type(4)));
typedef int v8i_t __attribute__((ext_vector_type(8)));

namespace gemm_i8 {

constexpr int WM = 4, WN = 2, BK = 64, BM = 64 * WM, BN = 64 * WN, NT = 32 * WM * WN, WS = BK + 16, CPR = BK / 16;
constexpr int CA = BM * CPR / NT, CB = BN * CPR / NT, RPS = NT / CPR;
constexpr int LDS = 2 * (BM + BN) * WS;  // 61440 bytes: WS = 80 makes 16 lanes' 16-byte reads hit 64 different banks
constexpr int G = 4;

#if defined(RDNA4)
static __device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Thread threadIdx.x stages the 16 bytes at k a_col() of tile rows a_row() + RPS i, i < CA, of the weights and of
// tokens t0 + a_row() + RPS i, i < CB.
static __device__ __forceinline__ int a_row() {
    return threadIdx.x / CPR;
}

static __device__ __forceinline__ int a_col() {
    return 16 * (threadIdx.x % CPR);
}

// acc[i][j]: lane (h = lane / 16, r = lane % 16) of wave (wm, wn) holds tile rows 64 wm + 16 i + 8 h + e (e = 0..7) of
// token t0 + 64 wn + 16 j + r. Token rows past T read row T - 1 and must be dropped by the caller. asrc is this thread's
// weight row for tile row a_row() at k a_col(), astep the distance to its next one (tile row + RPS). The k blocks run
// from k0 around to k0 - 1 (a start staggered by block runs the long k loop of the FFN down linear ~8% faster).
static __device__ __forceinline__ void mainloop(const int8_t * __restrict__ asrc, const size_t astep,
                                                const int8_t * __restrict__ X, const int T, const int K, const int t0,
                                                const int k0, int8_t * smem, v8i_t (&acc)[4][4]) {
    int8_t * sa = smem;                // [2][BM * WS]
    int8_t * sb = smem + 2 * BM * WS;  // [2][BN * WS]
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int wm = wave % WM, wn = wave / WM, h = lane / 16, r = lane % 16;
    const int8_t * bsrc[CB];
#pragma unroll
    for (int i = 0; i < CB; ++i) {
        bsrc[i] = X + (size_t) min(t0 + a_row() + RPS * i, T - 1) * K + a_col();
    }
    const int ldst = a_row() * WS + a_col();
    const int nk   = K / BK;
    int       kn   = k0;  // the k block loaded next

#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = v8i_t{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }
    v4i_t ra[CA], rb[CB];
#pragma unroll
    for (int i = 0; i < CA; ++i) {
        ra[i] = *(const v4i_t *) (asrc + astep * i + BK * kn);
    }
#pragma unroll
    for (int i = 0; i < CB; ++i) {
        rb[i] = *(const v4i_t *) (bsrc[i] + BK * kn);
    }
#pragma unroll
    for (int i = 0; i < CA; ++i) {
        *(v4i_t *) &sa[ldst + RPS * i * WS] = ra[i];
    }
#pragma unroll
    for (int i = 0; i < CB; ++i) {
        *(v4i_t *) &sb[ldst + RPS * i * WS] = rb[i];
    }
    lds_barrier();

    for (int kb = 0; kb < nk; ++kb) {
        const int buf = kb & 1;
        kn = kn + 1 == nk ? 0 : kn + 1;  // the last trip reloads the first block so the loop stays branch-free
#pragma unroll
        for (int i = 0; i < CA; ++i) {
            ra[i] = *(const v4i_t *) (asrc + astep * i + BK * kn);
        }
#pragma unroll
        for (int i = 0; i < CB; ++i) {
            rb[i] = *(const v4i_t *) (bsrc[i] + BK * kn);
        }
        __builtin_amdgcn_sched_barrier(0);
        const int8_t * ca = sa + buf * BM * WS;
        const int8_t * cb = sb + buf * BN * WS;
#pragma unroll
        for (int q = 0; q < BK / 32; ++q) {
            v4i_t a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = *(const v4i_t *) &ca[(64 * wm + 16 * i + r) * WS + BK / 2 * h + 16 * q];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                b[j] = *(const v4i_t *) &cb[(64 * wn + 16 * j + r) * WS + BK / 2 * h + 16 * q];
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const v2i_t af = s == 0 ? v2i_t{a[i].x, a[i].y} : v2i_t{a[i].z, a[i].w};
                        const v2i_t bf = s == 0 ? v2i_t{b[j].x, b[j].y} : v2i_t{b[j].z, b[j].w};
                        acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, af, true, bf, acc[i][j], false);
                    }
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        int8_t * na = sa + (buf ^ 1) * BM * WS;
        int8_t * nb = sb + (buf ^ 1) * BN * WS;
#pragma unroll
        for (int i = 0; i < CA; ++i) {
            *(v4i_t *) &na[ldst + RPS * i * WS] = ra[i];
        }
#pragma unroll
        for (int i = 0; i < CB; ++i) {
            *(v4i_t *) &nb[ldst + RPS * i * WS] = rb[i];
        }
        lds_barrier();
    }
}

// The plain weight rows of tile rows f0 + a_row() + RPS i.
static __device__ __forceinline__ const int8_t * weight_rows(const int8_t * W, const int K, const int f0) {
    return W + (size_t) (f0 + a_row()) * K + a_col();
}

static __device__ __forceinline__ void block_origin(const int N, const int T, int & f0, int & t0) {
    const int nf = N / BM, nt = (T + BN - 1) / BN, per = G * nf;
    const int group = blockIdx.x / per, first = group * G, gs = min(nt - first, G), in = blockIdx.x % per;
    f0 = (in / gs) * BM;
    t0 = (first + in % gs) * BN;
}

// dit-i8.cu's dequantize_i32 with the scale and bias at hand. contract(off): there the multiply and the add sit behind
// the nullptr checks in separate blocks and never fuse into an FMA, so they must not here either.
static __device__ __forceinline__ float dequant(const int32_t sum, const float act, const bool has_ws, const float ws,
                                                const bool has_b, const float b) {
#pragma clang fp contract(off)
    float value = (float) sum * act;
    if (has_ws) {
        value *= ws;
    }
    if (has_b) {
        value += b;
    }
    return value;
}

static __device__ __forceinline__ void unpack8(const float * p, float (&v)[8]) {
    const float4 a = *(const float4 *) p;
    const float4 b = *(const float4 *) (p + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
#endif // defined(RDNA4)

} // namespace gemm_i8

// The 256 features of a block are two heads of q, k or v (heads * 128 is a multiple of 256). v rows are cast straight
// from the accumulators; q and k rows go through LDS 32 tokens at a time so that one wave owns a (head, token) row
// and ggml_cuda_rms_norm_mul_rope_row sums its squares in the prologue kernel's order.
template <bool interleaved>
static __global__ void __launch_bounds__(gemm_i8::NT) gemm_i8_qkv_attn_prologue_kernel(
        const int8_t * __restrict__ W, const int8_t * __restrict__ X, const int T, const int K,
        const float * __restrict__ act_scales, const float * __restrict__ weight_scales, const float * __restrict__ bias,
        const float * __restrict__ q_w, const float * __restrict__ k_w, const float q_eps, const float k_eps,
        const float * __restrict__ pe, const int64_t pe_s0, const int64_t pe_s1, const int64_t pe_s2, const int64_t pe_s3,
        const int heads, float * __restrict__ q, half * __restrict__ k, half * __restrict__ v) {
#if defined(RDNA4)
    using namespace gemm_i8;
    constexpr int head_dim = 128, CT = 32, SS = BM + 4;  // tokens per LDS chunk, padded stage row (floats)
    constexpr int PE = head_dim;                          // per token: (cos, sin) of the 64 pairs
    __shared__ __attribute__((aligned(16))) int8_t smem[LDS];
    const int dim = heads * head_dim, N = 3 * dim;
    int f0, t0;
    block_origin(N, T, f0, t0);
    v8i_t acc[4][4];
    mainloop(weight_rows(W, K, f0), (size_t) RPS * K, X, T, K, t0, 0, smem, acc);

    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, wm = wave % WM, wn = wave / WM, h = lane / 16, r = lane % 16;
    const int part = f0 / dim, head0 = (f0 - part * dim) / head_dim;

    // block constants in LDS (the main loop ends on a barrier): weight scale and bias per feature, act scale per token
    float * stage  = (float *) smem;  // [CT][SS]
    float * pe_lds = stage + CT * SS;  // [CT][PE], non-interleaved rope only
    float * cw     = pe_lds + CT * PE;
    float * cb     = cw + BM;
    float * ca     = cb + BM;
    static_assert((CT * SS + CT * PE + 2 * BM + BN) * 4 <= LDS, "the epilogue fits in the main loop's LDS");
    const bool has_ws = weight_scales != nullptr, has_b = bias != nullptr;
    cw[threadIdx.x] = has_ws ? weight_scales[f0 + threadIdx.x] : 1.0f;
    cb[threadIdx.x] = has_b ? bias[f0 + threadIdx.x] : 0.0f;
    if (threadIdx.x < BN) {
        ca[threadIdx.x] = act_scales[min(t0 + (int) threadIdx.x, T - 1)];
    }

    if (part == 2) {
        lds_barrier();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int fl = 64 * wm + 16 * i + 8 * h, head = head0 + fl / head_dim;
            float wsv[8], bv[8];
            unpack8(cw + fl, wsv);
            unpack8(cb + fl, bv);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int t = t0 + 64 * wn + 16 * j + r;
                if (t >= T) {
                    continue;
                }
                const float act = ca[64 * wn + 16 * j + r];
                half o[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    o[e] = ggml_cuda_cast<half>(dequant(acc[i][j][e], act, has_ws, wsv[e], has_b, bv[e]));
                }
                *(v4i_t *) (v + ((size_t) head * T + t) * head_dim + fl % head_dim) = *(const v4i_t *) o;
            }
        }
        return;
    }
    const float * qk_w = part == 0 ? q_w : k_w;
    const float   eps  = part == 0 ? q_eps : k_eps;
#pragma unroll 1
    for (int g = 0; g < 2; ++g) {  // token half 64 g .. 64 g + 63, staged by the waves with wn == g
#pragma unroll                     // constant acc indices (a runtime index puts acc in scratch)
        for (int hh = 0; hh < 2; ++hh) {
            const int c = 2 * g + hh;  // chunk: tokens CT c .. CT c + 31 of the block
            // this chunk's pe, loaded while the stage fills: (cos, sin) of pair p at pe_lds[tl * PE + 2 p]
            float pv[CT * PE / NT];
            if (!interleaved) {
#pragma unroll
                for (int u = 0; u < CT * PE / NT; ++u) {
                    const int idx = threadIdx.x + NT * u, tl = idx / PE, p = (idx % PE) / 2, cs = idx % 2;
                    const int t = min(t0 + CT * c + tl, T - 1);
                    pv[u] = pe[t * pe_s3 + p * pe_s2 + cs * pe_s1];
                }
            }
            lds_barrier();
            if (wn == g) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int fl = 64 * wm + 16 * i + 8 * h;
                    float wsv[8], bv[8];
                    unpack8(cw + fl, wsv);
                    unpack8(cb + fl, bv);
#pragma unroll
                    for (int jj = 0; jj < 2; ++jj) {
                        const int j = 2 * hh + jj, tl = 16 * jj + r;
                        const float act = ca[64 * g + 16 * j + r];
                        float o[8];
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            o[e] = dequant(acc[i][j][e], act, has_ws, wsv[e], has_b, bv[e]);
                        }
                        *(float4 *) &stage[tl * SS + fl]     = make_float4(o[0], o[1], o[2], o[3]);
                        *(float4 *) &stage[tl * SS + fl + 4] = make_float4(o[4], o[5], o[6], o[7]);
                    }
                }
            }
            if (!interleaved) {
#pragma unroll
                for (int u = 0; u < CT * PE / NT; ++u) {
                    pe_lds[threadIdx.x + NT * u] = pv[u];
                }
            }
            lds_barrier();
#pragma unroll 2
            for (int row = wave; row < 2 * CT; row += NT / 32) {
                const int tl = row % CT, hl = row / CT, t = t0 + CT * c + tl;
                if (t >= T) {
                    continue;
                }
                const int     head   = head0 + hl;
                const float * x      = stage + tl * SS + head_dim * hl;
                const auto    load   = [&](const int64_t i) { return x[i]; };
                const float * pe_row = interleaved ? pe + t * pe_s3 : pe_lds + tl * PE;
                const int64_t s0 = interleaved ? pe_s0 : 0, s1 = interleaved ? pe_s1 : 1, s2 = interleaved ? pe_s2 : 2;
                if (part == 0) {
                    ggml_cuda_rms_norm_mul_rope_row<interleaved, float>(load, qk_w, pe_row, s0, s1, s2, head_dim / 2, eps,
                                                                        q + ((size_t) head * T + t) * head_dim);
                } else {
                    ggml_cuda_rms_norm_mul_rope_row<interleaved, half>(load, qk_w, pe_row, s0, s1, s2, head_dim / 2, eps,
                                                                       k + ((size_t) head * T + t) * head_dim);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(W, X, T, K, act_scales, weight_scales, bias, q_w, k_w, q_eps, k_eps, pe, pe_s0, pe_s1, pe_s2, pe_s3,
                     heads, q, k, v);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

// mul_add_i32_kernel's arithmetic straight from the accumulators. Each lane reads the 8 residual values it writes, all
// four of a token's before any store, since res and dst may be the same tensor.
static __global__ void __launch_bounds__(gemm_i8::NT) gemm_i8_mul_add_kernel(
        const int8_t * __restrict__ W, const int8_t * __restrict__ X, const int N, const int T, const int K, const int S,
        const float * __restrict__ act_scales, const float * __restrict__ weight_scales, const float * __restrict__ bias,
        const float * __restrict__ gate, const float * res, float * dst) {
#if defined(RDNA4)
    using namespace gemm_i8;
    __shared__ __attribute__((aligned(16))) int8_t smem[LDS];
    int f0, t0;
    block_origin(N, T, f0, t0);
    v8i_t acc[4][4];
    mainloop(weight_rows(W, K, f0), (size_t) RPS * K, X, T, K, t0, (int) (blockIdx.x * S % (K / BK)), smem, acc);

    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, wm = wave % WM, wn = wave / WM, h = lane / 16, r = lane % 16;
    // block constants in LDS (the main loop ends on a barrier): weight scale, bias and gate per feature, act scale per token
    float * cw = (float *) smem;
    float * cb = cw + BM;
    float * cg = cb + BM;
    float * ca = cg + BM;
    const bool has_ws = weight_scales != nullptr, has_b = bias != nullptr;
    cw[threadIdx.x] = has_ws ? weight_scales[f0 + threadIdx.x] : 1.0f;
    cb[threadIdx.x] = has_b ? bias[f0 + threadIdx.x] : 0.0f;
    cg[threadIdx.x] = gate[f0 + threadIdx.x];
    if (threadIdx.x < BN) {
        ca[threadIdx.x] = act_scales[min(t0 + (int) threadIdx.x, T - 1)];
    }
    lds_barrier();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = t0 + 64 * wn + 16 * j + r;
        if (t >= T) {
            continue;
        }
        const float act = ca[64 * wn + 16 * j + r];
        float rv[4][8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            unpack8(res + (size_t) t * N + f0 + 64 * wm + 16 * i + 8 * h, rv[i]);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int fl = 64 * wm + 16 * i + 8 * h;
            float wsv[8], bv[8], gv[8], o[8];
            unpack8(cw + fl, wsv);
            unpack8(cb + fl, bv);
            unpack8(cg + fl, gv);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                o[e] = __fadd_rn(rv[i][e], __fmul_rn(dequant(acc[i][j][e], act, has_ws, wsv[e], has_b, bv[e]), gv[e]));
            }
            float * out = dst + (size_t) t * N + f0 + fl;
            *(float4 *) out       = make_float4(o[0], o[1], o[2], o[3]);
            *(float4 *) (out + 4) = make_float4(o[4], o[5], o[6], o[7]);
        }
    }
#else
    GGML_UNUSED_VARS(W, X, N, T, K, S, act_scales, weight_scales, bias, gate, res, dst);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

static bool gemm_i8_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_I8_GEMM");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

bool ggml_cuda_gemm_i8_supported(const int cc, const int64_t n, const int64_t k) {
    return gemm_i8_enabled() && GGML_CUDA_CC_IS_RDNA4(cc) && n % gemm_i8::BM == 0 && k % gemm_i8::BK == 0 &&
           k <= INT32_MAX / 4;
}

static int64_t gemm_i8_blocks(const int64_t n, const int64_t rows) {
    return (n / gemm_i8::BM) * ((rows + gemm_i8::BN - 1) / gemm_i8::BN);
}

void ggml_cuda_gemm_i8_qkv_attn_prologue(
        const int8_t * w, const int8_t * x, const float * act_scales, const float * weight_scales, const float * bias,
        const int64_t rows, const int64_t k, const float * q_w, const float * k_w, const float q_eps, const float k_eps,
        const float * pe, const int64_t pe_s0, const int64_t pe_s1, const int64_t pe_s2, const int64_t pe_s3,
        const bool interleaved, const int64_t heads, float * q, half * k_out, half * v, cudaStream_t stream) {
    const auto kernel = interleaved ? gemm_i8_qkv_attn_prologue_kernel<true> : gemm_i8_qkv_attn_prologue_kernel<false>;
    kernel<<<gemm_i8_blocks(3 * heads * 128, rows), gemm_i8::NT, 0, stream>>>(
        w, x, (int) rows, (int) k, act_scales, weight_scales, bias, q_w, k_w, q_eps, k_eps, pe, pe_s0, pe_s1, pe_s2, pe_s3,
        (int) heads, q, k_out, v);
}

void ggml_cuda_gemm_i8_mul_add(
        const int8_t * w, const int8_t * x, const float * act_scales, const float * weight_scales, const float * bias,
        const int64_t rows, const int64_t k, const int64_t n, const float * gate, const float * res, float * dst,
        cudaStream_t stream) {
    const int stagger = k >= 8192 ? 5 : 0;  // k blocks staggered by 5 per block: only the long k loop gains
    gemm_i8_mul_add_kernel<<<gemm_i8_blocks(n, rows), gemm_i8::NT, 0, stream>>>(
        w, x, (int) n, (int) rows, (int) k, stagger, act_scales, weight_scales, bias, gate, res, dst);
}
