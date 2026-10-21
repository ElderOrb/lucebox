#include "blas.h"

#include <hipblaslt/hipblaslt-ext.hpp>

namespace luce::moss_image {

namespace {

constexpr size_t  kWorkspaceBytes = 64u << 20;
constexpr int     kCandidates     = 16;
constexpr int64_t kRowBucket      = 256;

struct TypeInfo {
    hipDataType          ab;
    hipDataType          d;
    hipblasComputeType_t compute;
    hipDataType          scale;
};

TypeInfo type_info(const GemmType type) {
    switch (type) {
        case GemmType::i8_i32:   return { HIP_R_8I, HIP_R_32I, HIPBLAS_COMPUTE_32I, HIP_R_32I };
        case GemmType::bf16_f32: return { HIP_R_16BF, HIP_R_32F, HIPBLAS_COMPUTE_32F, HIP_R_32F };
        case GemmType::f16_f32:  return { HIP_R_16F, HIP_R_32F, HIPBLAS_COMPUTE_32F, HIP_R_32F };
    }
    return {};
}

// alpha 1 and beta 0 in the problem's scale type.
const void * scalar(const GemmType type, const bool one) {
    static const int32_t i32[2] = { 0, 1 };
    static const float   f32[2] = { 0.0f, 1.0f };
    return type == GemmType::i8_i32 ? static_cast<const void *>(&i32[one]) : static_cast<const void *>(&f32[one]);
}

} // namespace

Blas::~Blas() {
    for (auto & [problem, plan] : plans_) {
        destroy_plan(plan);
    }
    if (workspace_ != nullptr) {
        (void) hipFree(workspace_);
    }
    if (handle_ != nullptr) {
        hipblasLtDestroy(handle_);
    }
}

bool Blas::init(hipStream_t stream) {
    stream_ = stream;
    return hipblasLtCreate(&handle_) == HIPBLAS_STATUS_SUCCESS && hipMalloc(&workspace_, kWorkspaceBytes) == hipSuccess;
}

bool Blas::make_plan(const Problem & p, Plan & plan) const {
    const TypeInfo           t       = type_info(p.type);
    const hipblasOperation_t trans_a = HIPBLAS_OP_T;
    return hipblasLtMatmulDescCreate(&plan.desc, t.compute, t.scale) == HIPBLAS_STATUS_SUCCESS &&
           hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) ==
               HIPBLAS_STATUS_SUCCESS &&
           hipblasLtMatrixLayoutCreate(&plan.a_layout, t.ab, p.k, p.n, p.lda) == HIPBLAS_STATUS_SUCCESS &&
           hipblasLtMatrixLayoutCreate(&plan.b_layout, t.ab, p.k, p.rows, p.ldb) == HIPBLAS_STATUS_SUCCESS &&
           hipblasLtMatrixLayoutCreate(&plan.d_layout, t.d, p.n, p.rows, p.ldd) == HIPBLAS_STATUS_SUCCESS;
}

void Blas::destroy_plan(Plan & plan) const {
    if (plan.d_layout != nullptr) {
        hipblasLtMatrixLayoutDestroy(plan.d_layout);
    }
    if (plan.b_layout != nullptr) {
        hipblasLtMatrixLayoutDestroy(plan.b_layout);
    }
    if (plan.a_layout != nullptr) {
        hipblasLtMatrixLayoutDestroy(plan.a_layout);
    }
    if (plan.desc != nullptr) {
        hipblasLtMatmulDescDestroy(plan.desc);
    }
    plan = Plan{};
}

// The bucket's kernel when it runs this exact problem, else the fastest of the heuristic's candidates, timed on the
// call's own operands (each candidate writes the same product to d). False inside stream capture, which cannot time.
bool Blas::choose_algo(const Problem & p, Plan & plan, const void * a, const void * b, void * d) {
    const void * alpha = scalar(p.type, true);
    const void * beta  = scalar(p.type, false);
    Problem bucket = p;
    bucket.rows    = (p.rows + kRowBucket - 1) / kRowBucket * kRowBucket;
    const auto known = bucket_algos_.find(bucket);
    if (known != bucket_algos_.end()) {
        hipblasLtMatmulAlgo_t algo           = known->second;
        size_t                workspace_size = 0;
        if (hipblaslt_ext::matmulIsAlgoSupported(handle_, plan.desc, alpha, plan.a_layout, plan.b_layout, beta,
                                                 plan.d_layout, plan.d_layout, algo,
                                                 workspace_size) == HIPBLAS_STATUS_SUCCESS &&
            workspace_size <= kWorkspaceBytes) {
            plan.algo  = algo;
            plan.tuned = true;
            return true;
        }
    }
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream_, &capture) != hipSuccess || capture != hipStreamCaptureStatusNone) {
        return false;
    }

    hipblasLtMatmulPreference_t pref = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) {
        return true;
    }
    const uint64_t max_workspace = kWorkspaceBytes;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace,
                                          sizeof(max_workspace));
    hipblasLtMatmulHeuristicResult_t candidates[kCandidates];
    int                              n_found = 0;
    const hipblasStatus_t            status =
        hipblasLtMatmulAlgoGetHeuristic(handle_, plan.desc, plan.a_layout, plan.b_layout, plan.d_layout,
                                        plan.d_layout, pref, kCandidates, candidates, &n_found);
    hipblasLtMatmulPreferenceDestroy(pref);
    if (status != HIPBLAS_STATUS_SUCCESS || n_found == 0) {
        return true;
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop  = nullptr;
    if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess) {
        return true;
    }
    int   best    = -1;
    float best_ms = 0.0f;
    for (int i = 0; i < n_found; ++i) {
        if (candidates[i].workspaceSize > kWorkspaceBytes) {
            continue;
        }
        const auto run = [&]() {
            return hipblasLtMatmul(handle_, plan.desc, alpha, a, plan.a_layout, b, plan.b_layout, beta, d,
                                   plan.d_layout, d, plan.d_layout, &candidates[i].algo, workspace_, kWorkspaceBytes,
                                   stream_);
        };
        if (run() != HIPBLAS_STATUS_SUCCESS) {
            continue;
        }
        float ms = 0.0f;
        if (hipEventRecord(start, stream_) != hipSuccess || run() != HIPBLAS_STATUS_SUCCESS ||
            run() != HIPBLAS_STATUS_SUCCESS || hipEventRecord(stop, stream_) != hipSuccess ||
            hipEventSynchronize(stop) != hipSuccess || hipEventElapsedTime(&ms, start, stop) != hipSuccess) {
            continue;
        }
        if (best < 0 || ms < best_ms) {
            best    = i;
            best_ms = ms;
        }
    }
    (void) hipEventDestroy(start);
    (void) hipEventDestroy(stop);
    if (best >= 0) {
        plan.algo             = candidates[best].algo;
        plan.tuned            = true;
        bucket_algos_[bucket] = plan.algo;
    }
    return true;
}

bool Blas::gemm(const GemmType type, const void * a, const int64_t lda, const void * b, const int64_t ldb, void * d,
                const int64_t ldd, const int64_t n, const int64_t rows, const int64_t k) {
    const Problem p{ type, n, rows, k, lda, ldb, ldd };
    auto          it = plans_.find(p);
    Plan          untimed;  // a plan made inside stream capture, used once and not kept
    Plan *        plan = nullptr;
    if (it != plans_.end()) {
        plan = &it->second;
    } else {
        Plan made;
        if (!make_plan(p, made)) {
            destroy_plan(made);
            return false;
        }
        if (choose_algo(p, made, a, b, d)) {
            plan = &(plans_[p] = made);
        } else {
            untimed = made;
            plan    = &untimed;
        }
    }
    const hipblasStatus_t status =
        hipblasLtMatmul(handle_, plan->desc, scalar(type, true), a, plan->a_layout, b, plan->b_layout,
                        scalar(type, false), d, plan->d_layout, d, plan->d_layout, plan->tuned ? &plan->algo : nullptr,
                        workspace_, plan->tuned ? kWorkspaceBytes : 0, stream_);
    if (plan == &untimed) {
        destroy_plan(untimed);
    }
    return status == HIPBLAS_STATUS_SUCCESS;
}

} // namespace luce::moss_image
