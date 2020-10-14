#pragma once
#include <bits/stdc++.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

class OC_gemm {
private:
    const int M, N, K;
    int tm, tn, tk;

    const int num_stream;
    cudaStream_t *streams;
    cublasHandle_t *handles;

    half **A_tiles, **B_tiles;
    float **fA_tiles, **fB_tiles, **C_tiles;

    void tile_size();
    size_t dev_mem_per_stream;

public:
    OC_gemm() = delete;
    OC_gemm(const OC_gemm &o) = delete;
    void operator=(const OC_gemm &o) = delete;
    OC_gemm(int _M, int _N, int _K, int _num_stream = 4);
    ~OC_gemm();
    void gemm(cublasOperation_t transa, cublasOperation_t transb, const float &alpha, const half *A,
              int lda, const half *B, int ldb, const float &beta, float *C, int ldc);
    void gemm(cublasOperation_t transa, cublasOperation_t transb, const float &alpha,
              const float *A, int lda, const float *B, int ldb, const float &beta, float *C,
              int ldc);
};