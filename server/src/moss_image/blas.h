// hipBLASLt GEMMs of MOSS-Image: every linear and the VAE attention products.
//
// All products are D[rows][n] = B[rows][k] * A[n][k]^T in row-major terms: A holds the weights (or the keys),
// one row per output column, B the activations, one row per token. hipBLASLt sees column-major D (n x rows) =
// A^T (n x k) * B (k x rows) with TRANSA = T.
//
// hipBLASLt's implicit heuristic is not its fastest kernel for every shape: the first product of a shape times the
// heuristic's top 16 candidates and keeps the fastest. The token count moves with every prompt while the best kernel
// does not, so one tuning serves a 256-row bucket and each new exact problem only checks the bucket's kernel.

#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <map>
#include <tuple>

namespace luce::moss_image {

enum class GemmType {
    i8_i32,    // int8 A and B, int32 D
    bf16_f32,  // bf16 A and B, f32 accumulation and D
    f16_f32,   // f16 A and B, f32 accumulation and D
};

class Blas {
public:
    Blas() = default;
    ~Blas();
    Blas(const Blas &) = delete;
    Blas & operator=(const Blas &) = delete;

    // The handle and the workspace; every product runs on `stream`.
    bool init(hipStream_t stream);

    // D = B * A^T with leading dimensions (row strides, in elements) lda >= k, ldb >= k, ldd >= n.
    bool gemm(GemmType type, const void * a, int64_t lda, const void * b, int64_t ldb, void * d, int64_t ldd,
              int64_t n, int64_t rows, int64_t k);

private:
    struct Problem {
        GemmType type;
        int64_t  n, rows, k, lda, ldb, ldd;

        bool operator<(const Problem & o) const {
            return std::tie(type, n, rows, k, lda, ldb, ldd) < std::tie(o.type, o.n, o.rows, o.k, o.lda, o.ldb, o.ldd);
        }
    };

    // A problem's descriptors and kernel, made on its first product.
    struct Plan {
        hipblasLtMatmulDesc_t   desc     = nullptr;
        hipblasLtMatrixLayout_t a_layout = nullptr;
        hipblasLtMatrixLayout_t b_layout = nullptr;
        hipblasLtMatrixLayout_t d_layout = nullptr;
        hipblasLtMatmulAlgo_t   algo     = {};
        bool                    tuned    = false;  // false: hipBLASLt's implicit heuristic
    };

    bool make_plan(const Problem & p, Plan & plan) const;
    void destroy_plan(Plan & plan) const;
    bool choose_algo(const Problem & p, Plan & plan, const void * a, const void * b, void * d);

    hipStream_t       stream_    = nullptr;
    hipblasLtHandle_t handle_    = nullptr;
    void *            workspace_ = nullptr;
    std::map<Problem, Plan>                  plans_;
    std::map<Problem, hipblasLtMatmulAlgo_t> bucket_algos_;  // the tuned kernel of each 256-row bucket
};

} // namespace luce::moss_image
