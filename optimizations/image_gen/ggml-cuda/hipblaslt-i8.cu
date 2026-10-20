#include "hipblaslt-i8.cuh"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_HIPBLASLT)
#include <hipblaslt/hipblaslt-ext.hpp>

#include <map>
#include <mutex>
#include <tuple>

// Workspace a candidate kernel may use.
static constexpr size_t workspace_bytes = 64u << 20;

static hipblasLtHandle_t hipblaslt_handle(const int device) {
    static hipblasLtHandle_t handles[GGML_CUDA_MAX_DEVICES] = {};
    if (handles[device] == nullptr) {
        CUBLAS_CHECK(hipblasLtCreate(&handles[device]));
    }
    return handles[device];
}

// hipBLASLt's implicit heuristic is not its fastest kernel for every shape: time up to 16 candidates on first use and
// keep the fastest per device and shape. The token count n moves with every prompt length while the best kernel does
// not, so one tuning serves a 256-token bucket and each new exact problem only checks the bucket's kernel. Inside
// stream capture an untuned bucket falls back to the implicit heuristic (null).
static const hipblasLtMatmulAlgo_t * best_algo(
        ggml_backend_cuda_context & ctx, hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc,
        hipblasLtMatrixLayout_t a_desc, hipblasLtMatrixLayout_t b_desc, hipblasLtMatrixLayout_t d_desc,
        const void * alpha, const void * a, const void * b, const void * beta, void * d,
        int64_t m, int64_t n, int64_t k, void * workspace) {
    using shape = std::tuple<int, int64_t, int64_t, int64_t>;
    static std::mutex mutex;
    static std::map<shape, hipblasLtMatmulAlgo_t> best;         // exact problems, checked or tuned
    static std::map<shape, hipblasLtMatmulAlgo_t> bucket_best;  // tuned kernel per 256-token bucket
    std::lock_guard<std::mutex> lock(mutex);
    const shape key        = std::make_tuple(ctx.device, m, n, k);
    const shape bucket_key = std::make_tuple(ctx.device, m, GGML_PAD(n, 256), k);
    auto it = best.find(key);
    if (it != best.end()) {
        return &it->second;
    }
    auto bucket_it = bucket_best.find(bucket_key);
    if (bucket_it != bucket_best.end()) {
        hipblasLtMatmulAlgo_t algo = bucket_it->second;
        size_t workspace_size = 0;
        if (hipblaslt_ext::matmulIsAlgoSupported(handle, desc, alpha, a_desc, b_desc, beta, d_desc, d_desc, algo,
                                                 workspace_size) == HIPBLAS_STATUS_SUCCESS &&
                workspace_size <= workspace_bytes) {
            return &(best[key] = algo);
        }
    }
    cudaStream_t stream = ctx.stream();
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    CUDA_CHECK(hipStreamIsCapturing(stream, &capture));
    if (capture != hipStreamCaptureStatusNone) {
        return nullptr;
    }
    hipblasLtMatmulPreference_t pref;
    CUBLAS_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    const uint64_t max_workspace = workspace_bytes;
    CUBLAS_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace,
                                                       sizeof(max_workspace)));
    hipblasLtMatmulHeuristicResult_t heuristics[16];
    int n_found = 0;
    const hipblasStatus_t status = hipblasLtMatmulAlgoGetHeuristic(handle, desc, a_desc, b_desc, d_desc, d_desc, pref,
                                                                   16, heuristics, &n_found);
    CUBLAS_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
    if (status != HIPBLAS_STATUS_SUCCESS || n_found == 0) {
        return nullptr;
    }
    hipEvent_t start, stop;
    CUDA_CHECK(hipEventCreate(&start));
    CUDA_CHECK(hipEventCreate(&stop));
    int   best_index = -1;
    float best_ms    = 0.0f;
    for (int i = 0; i < n_found; ++i) {
        if (heuristics[i].workspaceSize > workspace_bytes) {
            continue;
        }
        const auto run = [&]() {
            return hipblasLtMatmul(handle, desc, alpha, a, a_desc, b, b_desc, beta, d, d_desc, d, d_desc,
                                   &heuristics[i].algo, workspace, workspace_bytes, stream);
        };
        if (run() != HIPBLAS_STATUS_SUCCESS) {
            continue;
        }
        CUDA_CHECK(hipEventRecord(start, stream));
        run();
        run();
        CUDA_CHECK(hipEventRecord(stop, stream));
        CUDA_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(hipEventElapsedTime(&ms, start, stop));
        if (best_index < 0 || ms < best_ms) {
            best_index = i;
            best_ms    = ms;
        }
    }
    CUDA_CHECK(hipEventDestroy(start));
    CUDA_CHECK(hipEventDestroy(stop));
    if (best_index < 0) {
        return nullptr;
    }
    bucket_best[bucket_key] = heuristics[best_index].algo;
    return &(best[key] = heuristics[best_index].algo);
}

bool ggml_cuda_gemm_i8_hipblaslt(ggml_backend_cuda_context & ctx, const int8_t * w, const int8_t * x, int32_t * sums,
                                 const int64_t n, const int64_t rows, const int64_t k) {
    hipblasLtHandle_t       handle = hipblaslt_handle(ctx.device);
    hipblasLtMatmulDesc_t   desc;
    hipblasLtMatrixLayout_t a_desc, b_desc, d_desc;
    const hipblasOperation_t trans_a = HIPBLAS_OP_T;
    CUBLAS_CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32I, HIP_R_32I));
    CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&a_desc, HIP_R_8I, k, n, k));
    CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&b_desc, HIP_R_8I, k, rows, k));
    CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&d_desc, HIP_R_32I, n, rows, n));
    const int32_t alpha = 1;
    const int32_t beta  = 0;
    ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool(), workspace_bytes);
    const hipblasLtMatmulAlgo_t * algo = best_algo(ctx, handle, desc, a_desc, b_desc, d_desc, &alpha, w, x, &beta, sums,
                                                   n, rows, k, workspace.get());
    const hipblasStatus_t status = hipblasLtMatmul(handle, desc, &alpha, w, a_desc, x, b_desc, &beta, sums, d_desc, sums,
                                                   d_desc, algo, workspace.get(), algo != nullptr ? workspace_bytes : 0,
                                                   ctx.stream());
    CUBLAS_CHECK(hipblasLtMatrixLayoutDestroy(d_desc));
    CUBLAS_CHECK(hipblasLtMatrixLayoutDestroy(b_desc));
    CUBLAS_CHECK(hipblasLtMatrixLayoutDestroy(a_desc));
    CUBLAS_CHECK(hipblasLtMatmulDescDestroy(desc));
    return status == HIPBLAS_STATUS_SUCCESS;
}
#else
bool ggml_cuda_gemm_i8_hipblaslt(ggml_backend_cuda_context & ctx, const int8_t * w, const int8_t * x, int32_t * sums,
                                 const int64_t n, const int64_t rows, const int64_t k) {
    GGML_UNUSED_VARS(ctx, w, x, sums, n, rows, k);
    return false;
}
#endif // defined(GGML_USE_HIP) && defined(GGML_HIP_HIPBLASLT)
