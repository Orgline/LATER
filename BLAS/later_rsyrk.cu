#include "LATER.h"

#include <cuda_fp16.h>

#define BLOCKSIZE 2048

float pan = 0.0;
float ge = 0.0;



void syrk(cublasHandle_t handle, int n, int k, float *A, int lda, float *C, int ldc, __half *hwork)
{
    //printf("n = %d\n", n);
    //gpuErrchk( cudaPeekAtLastError() );
    float sone  = 1.0;
    float szero = 0.0;
    if(n<=BLOCKSIZE)
    {
        startTimer();
        cublasSsyrk(handle,
            CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            n, k,
            &sone,
            A, lda,
            &szero,
            C, ldc
        );
        pan+=stopTimer();
        return;
    }
    syrk(handle, n/2, k, A, lda, C, ldc, hwork);
    
    startTimer();
    __half *Ah = hwork;
    __half *Bh = hwork+n/2*k;

    dim3 grid((n/2+1)/32, (k+1)/32);
    dim3 block(32,32);
    s2h<<<grid, block>>>(n/2, k, A+n/2, lda, Ah, n/2);
    s2h<<<grid, block>>>(n/2, k, A, lda, Bh, n/2);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, n/2, n/2, k,
        &sone, Ah, CUDA_R_16F, n/2, Bh, CUDA_R_16F, n/2,
        &szero, C+n/2, CUDA_R_32F, ldc, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    float t = stopTimer();

    //printf("GEMM size m,n,k = %d %d %d,takes %lf, %lf TFLOPS\n", n/2, n/2, k, t, 2.0*n/2*n/2*k/1e9/t);

    ge+=t;

    syrk(handle, n/2, k, A+n/2, lda, C+n/2+ldc/2*n, ldc, hwork);

    //Transpose and cpy

}

void later_rsyrk(int n, int k, float *A, int lda, float *C, int ldc, __half *work)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    syrk(handle, n, k, A, lda, C, ldc, work);

    printf("Panel takes %lf ms\n Gemm takes %lf ms\n", pan, ge);

    cublasDestroy(handle);
}