#include "fattn-rdna4.cuh"

// Flash attention forward for RDNA4 WMMA, head size 128, no mask and no GQA: the long self-attention of diffusion
// transformers, ~1.5x the speed of the generic MMA kernel at 4,000+ tokens.
//
// A block takes 192 query rows of one head (12 waves x 16 rows) and walks the keys in tiles of 64. The scores are
// computed transposed, S^T = K Q^T, because in the gfx12 WMMA result layout every lane then holds one query with 8
// consecutive keys: the row max needs one exchange with lane ^ 16, the probabilities are already laid out as the B
// operand of O^T += V^T P^T, and rescaling the output is per lane. V is transposed into LDS for that product. Scores
// are in the log2 domain (Q is pre-scaled by scale * log2(e)), probabilities in f16, all sums in f32.

#if defined(GGML_USE_HIP)
namespace {

typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float    float8_t __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));

constexpr int D        = 128;
constexpr int WAVES    = 12;
constexpr int BQ       = 16 * WAVES;  // query rows per block
constexpr int BK       = 64;          // key rows per tile
constexpr int THREADS  = 32 * WAVES;
constexpr int K_STRIDE = D + 8;       // halves per K row in LDS, padded against bank conflicts
constexpr int V_STRIDE = BK + 8;      // halves per V^T row in LDS (one row per head dimension)

struct fattn_rdna4_params {
    const float *    Q;    // [D, Lq, H, N]
    const _Float16 * K;    // [D, Lk, H, N]
    const _Float16 * V;    // [D, Lk, H, N]
    float *          dst;  // [D, H, Lq, N], contiguous
    int              Lq;
    int              Lk;
    int              H;
    int64_t          q_s1, q_s2, q_s3;  // element strides of tokens, heads and sequences
    int64_t          k_s1, k_s2, k_s3;
    int64_t          v_s1, v_s2, v_s3;
    float            scale_log2;
};

__global__ void __launch_bounds__(THREADS) __attribute__((amdgpu_waves_per_eu(6, 6)))
fattn_rdna4_d128(const fattn_rdna4_params p) {
#if defined(RDNA4)
    __shared__ _Float16 sK[BK * K_STRIDE];
    __shared__ _Float16 sVt[D * V_STRIDE];

    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    const int hs   = lane / 16;  // the 8-wide half of a 16-wide WMMA k step this lane holds

    // Full query tiles first, the partial last tile of every head at the end of the grid: its waves past the last query
    // skip the compute, so those half-empty blocks fill the last round of the grid instead of adding one.
    int tile, h, n;
    {
        const int full   = p.Lq / BQ;
        const int b      = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        const int n_full = full * p.H * (int) gridDim.z;
        if (b < n_full) {
            tile = b % full;
            h    = b / full % p.H;
            n    = b / (full * p.H);
        } else {
            tile = full;
            h    = (b - n_full) % p.H;
            n    = (b - n_full) / p.H;
        }
    }
    const int  qi     = tile * BQ + wave * 16 + lane % 16;  // this lane's query
    const bool active = tile * BQ + wave * 16 < p.Lq;

    // Q^T as the B operand of S^T = K Q^T: query qi, head dimensions 16 s + 8 hs + v.
    half8_t qb[D / 16];
    {
        const float * q_row = p.Q + (int64_t) min(qi, p.Lq - 1) * p.q_s1 + (int64_t) h * p.q_s2 + (int64_t) n * p.q_s3;
#pragma unroll
        for (int s = 0; s < D / 16; ++s) {
            const float4 a = *(const float4 *) (q_row + 16 * s + 8 * hs);
            const float4 b = *(const float4 *) (q_row + 16 * s + 8 * hs + 4);
            const float  c = p.scale_log2;
            qb[s] = half8_t{ (_Float16) (a.x * c), (_Float16) (a.y * c), (_Float16) (a.z * c), (_Float16) (a.w * c),
                             (_Float16) (b.x * c), (_Float16) (b.y * c), (_Float16) (b.z * c), (_Float16) (b.w * c) };
        }
    }

    float8_t o[D / 16] = {};  // O^T: query qi, head dimensions 16 t + 8 hs + v
    float    m = -INFINITY;   // running max of row qi, the same in lanes qi and qi ^ 16
    float    l = 0.0f;        // running sum over this lane's half of the keys

    const _Float16 * k_head = p.K + (int64_t) h * p.k_s2 + (int64_t) n * p.k_s3;
    const _Float16 * v_head = p.V + (int64_t) h * p.v_s2 + (int64_t) n * p.v_s3;

    // A tile goes global -> registers -> shared memory, the next tile's loads issued before the current tile is
    // computed. K as 16-byte chunks of its rows, V transposed in 8 x 8 blocks (keys 8 rb.., head dimensions 8 db..)
    // with rb = t % 8, so the 16 lanes of a half wave store two whole V^T rows (no bank conflicts). k_next is a
    // native vector type: an array of uint4 structs gets promoted to LDS and copied through it.
    constexpr int K_CHUNKS     = BK * D / 8;
    constexpr int K_PER_THREAD = (K_CHUNKS + THREADS - 1) / THREADS;
    constexpr int V_BLOCKS     = (BK / 8) * (D / 8);
    static_assert(V_BLOCKS <= THREADS, "one V block per thread");
    u32x4_t k_next[K_PER_THREAD];
    half8_t v_next[8];
    const auto load_tile = [&](const int kv0) {
#pragma unroll
        for (int i = 0; i < K_PER_THREAD; ++i) {
            const int c = threadIdx.x + i * THREADS;
            if (c < K_CHUNKS) {
                const int kv = min(kv0 + c / (D / 8), p.Lk - 1);
                k_next[i] = *(const u32x4_t *) (k_head + (int64_t) kv * p.k_s1 + 8 * (c % (D / 8)));
            }
        }
        if (threadIdx.x < V_BLOCKS) {
            const int rb = threadIdx.x % (BK / 8);
            const int db = threadIdx.x / (BK / 8);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int kv = min(kv0 + 8 * rb + i, p.Lk - 1);
                v_next[i] = *(const half8_t *) (v_head + (int64_t) kv * p.v_s1 + 8 * db);
            }
        }
    };

    load_tile(0);
    for (int kv0 = 0; kv0 < p.Lk; kv0 += BK) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < K_PER_THREAD; ++i) {
            const int c = threadIdx.x + i * THREADS;
            if (c < K_CHUNKS) {
                *(u32x4_t *) (sK + c / (D / 8) * K_STRIDE + 8 * (c % (D / 8))) = k_next[i];
            }
        }
        if (threadIdx.x < V_BLOCKS) {
            const int rb = threadIdx.x % (BK / 8);
            const int db = threadIdx.x / (BK / 8);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                *(half8_t *) (sVt + (8 * db + j) * V_STRIDE + 8 * rb) =
                    half8_t{ v_next[0][j], v_next[1][j], v_next[2][j], v_next[3][j],
                             v_next[4][j], v_next[5][j], v_next[6][j], v_next[7][j] };
            }
        }
        __syncthreads();
        if (kv0 + BK < p.Lk) {
            load_tile(kv0 + BK);
        }

        if (!active) {
            continue;
        }

        // S^T in 4 chunks of 16 keys: this lane gets keys kv0 + 16 c + 8 hs + v of query qi. The WMMAs go s-major, so
        // consecutive ones write different chunks (each still sums s = 0..7 in order), and every step's K fragments
        // are read one step ahead; left to itself the compiler waits for each read right before its WMMA.
        float8_t st[BK / 16] = {};
        {
            const _Float16 * k_lane = sK + (lane % 16) * K_STRIDE + 8 * hs;
            half8_t          kf[2][BK / 16];
#pragma unroll
            for (int c = 0; c < BK / 16; ++c) {
                kf[0][c] = *(const half8_t *) (k_lane + 16 * c * K_STRIDE);
            }
#pragma unroll
            for (int s = 0; s < D / 16; ++s) {
                if (s + 1 < D / 16) {
#pragma unroll
                    for (int c = 0; c < BK / 16; ++c) {
                        kf[(s + 1) % 2][c] = *(const half8_t *) (k_lane + 16 * c * K_STRIDE + 16 * (s + 1));
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int c = 0; c < BK / 16; ++c) {
                    st[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf[s % 2][c], qb[s], st[c]);
                }
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        if (kv0 + BK > p.Lk) {
#pragma unroll
            for (int c = 0; c < BK / 16; ++c) {
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    if (kv0 + 16 * c + 8 * hs + v >= p.Lk) {
                        st[c][v] = -INFINITY;
                    }
                }
            }
        }

        float mx = m;
#pragma unroll
        for (int c = 0; c < BK / 16; ++c) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                mx = fmaxf(mx, st[c][v]);
            }
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float alpha = __builtin_amdgcn_exp2f(m - mx);
        m = mx;

        half8_t pb[BK / 16];  // P^T as the B operand of O^T += V^T P^T
        float   sum = 0.0f;
#pragma unroll
        for (int c = 0; c < BK / 16; ++c) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float e = __builtin_amdgcn_exp2f(st[c][v] - mx);
                sum += e;
                pb[c][v] = (_Float16) e;
            }
        }
        l = l * alpha + sum;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {  // once the maxima settle alpha is 1: x * 1 = x
#pragma unroll
            for (int t = 0; t < D / 16; ++t) {
                o[t] *= alpha;
            }
        }
        // O^T += V^T P^T in 8 groups of 4 WMMAs: key chunk c = g / 2, head-dimension tiles t = 4 (g % 2) + i. Each
        // o[t] still sums c = 0..3 in order; every group's V^T fragments are read one group ahead.
        {
            const _Float16 * v_lane = sVt + (lane % 16) * V_STRIDE + 8 * hs;
            half8_t          vf[2][4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                vf[0][i] = *(const half8_t *) (v_lane + 16 * i * V_STRIDE);
            }
#pragma unroll
            for (int g = 0; g < 2 * (BK / 16); ++g) {
                if (g + 1 < 2 * (BK / 16)) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int t = 4 * ((g + 1) % 2) + i;
                        vf[(g + 1) % 2][i] = *(const half8_t *) (v_lane + 16 * t * V_STRIDE + 16 * ((g + 1) / 2));
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int t = 4 * (g % 2) + i;
                    o[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vf[g % 2][i], pb[g / 2], o[t]);
                }
                __builtin_amdgcn_sched_barrier(0);
            }
        }
    }

    l += __shfl_xor(l, 16, 32);
    if (qi < p.Lq) {
        const float inv = 1.0f / l;
        float *     out = p.dst + (((int64_t) n * p.Lq + qi) * p.H + h) * D;
#pragma unroll
        for (int t = 0; t < D / 16; ++t) {
            const float8_t r = o[t] * inv;
            *(float4 *) (out + 16 * t + 8 * hs)     = make_float4(r[0], r[1], r[2], r[3]);
            *(float4 *) (out + 16 * t + 8 * hs + 4) = make_float4(r[4], r[5], r[6], r[7]);
        }
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // defined(RDNA4)
}

}  // namespace
#endif // defined(GGML_USE_HIP)

bool ggml_cuda_flash_attn_ext_rdna4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    float scale, max_bias, logit_softcap;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const auto aligned16 = [](const ggml_tensor * t) {
        return (uintptr_t) t->data % 16 == 0 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
    };
    if (!GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ctx.device].cc) || dst->src[3] != nullptr ||
            (GGML_MAX_SRC > 4 && dst->src[4] != nullptr) || max_bias != 0.0f || logit_softcap != 0.0f ||
            Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32 ||
            Q->ne[0] != D || K->ne[0] != D || V->ne[0] != D || K->ne[1] != V->ne[1] || Q->ne[1] < 64 ||
            Q->ne[2] != K->ne[2] || V->ne[2] != K->ne[2] || Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3] ||
            Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) ||
            !aligned16(Q) || !aligned16(K) || !aligned16(V) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const fattn_rdna4_params p = {
        (const float *) Q->data, (const _Float16 *) K->data, (const _Float16 *) V->data, (float *) dst->data,
        (int) Q->ne[1], (int) K->ne[1], (int) Q->ne[2],
        (int64_t) (Q->nb[1] / sizeof(float)), (int64_t) (Q->nb[2] / sizeof(float)), (int64_t) (Q->nb[3] / sizeof(float)),
        (int64_t) (K->nb[1] / sizeof(half)), (int64_t) (K->nb[2] / sizeof(half)), (int64_t) (K->nb[3] / sizeof(half)),
        (int64_t) (V->nb[1] / sizeof(half)), (int64_t) (V->nb[2] / sizeof(half)), (int64_t) (V->nb[3] / sizeof(half)),
        scale * 1.4426950408889634f,
    };
    const dim3 grid((Q->ne[1] + BQ - 1) / BQ, Q->ne[2], Q->ne[3]);
    fattn_rdna4_d128<<<grid, THREADS, 0, ctx.stream()>>>(p);
    return true;
#else
    GGML_UNUSED_VARS(ctx, dst);
    return false;
#endif // defined(GGML_USE_HIP)
}
