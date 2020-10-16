#pragma once
#include "mem_pool.h"
#include <bits/stdc++.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
class OC_gemm {
private:
    template <typename T> using arr_t = std::unique_ptr<T[]>;
    const int M, N, K;
    int tm, tn, tk;

    const int num_stream;
    arr_t<cudaStream_t> streams;
    arr_t<cublasHandle_t> handles;

    arr_t<half *> A_tiles, B_tiles;
    arr_t<float *> fA_tiles, fB_tiles, C_tiles;

    void tile_size();

    std::shared_ptr<Mem_pool> pool;
    size_t mem_limit;

public:
    OC_gemm() = delete;
    OC_gemm(const OC_gemm &o) = delete;
    void operator=(const OC_gemm &o) = delete;
    OC_gemm(int _M, int _N, int _K, std::shared_ptr<Mem_pool> _pool, size_t _mem_limit = 0,
            int _num_stream = 4);
    ~OC_gemm();
    void gemm(cublasOperation_t transa, cublasOperation_t transb, const float &alpha, const half *A,
              int lda, const half *B, int ldb, const float &beta, float *C, int ldc);
    void gemm(cublasOperation_t transa, cublasOperation_t transb, const float &alpha,
              const float *A, int lda, const float *B, int ldb, const float &beta, float *C,
              int ldc);
};