#include "LATER.h"

#include <cuda_fp16.h>

void trsm(cublasHandle_t handle, int m, int n, float* A, int lda, float* B, int ldb)

void later_rtrsm(int m, int n, float* A, int lda, float* B, int ldb)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    return;
}