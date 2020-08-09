#include "LATER.h"

#include <cuda_fp16.h>

#define BLOCKSIZE 256

float sone = 1.0;
float snegone = -1.0;
float szero = 0.0;

float panelTime = 0.0;
float gemmTime = 0.0;

void trsm(cublasHandle_t handle, int m, int n, float* A, int lda, float* B, int ldb, __half* hwork)
{
    //printf("m,n=%d,%d\n", m, n);
    if(m <= BLOCKSIZE)
    {
        startTimer();
        cublasStrsm(handle,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            m, n, &sone,
            A, lda,
            B, ldb
        );
        panelTime += stopTimer();
        //printf("%lf\n",panelTime);
        return;
    }
    trsm(handle, m/2, n, A, lda, B, ldb, hwork);
    startTimer();
    
    __half *Ah = hwork;
    __half *Bh = hwork+m/2*m/2;

    dim3 grid((m/2+31)/32, (m/2+31)/32);
    dim3 block(32,32);
    s2h<<<grid, block>>>(m/2, m/2, A+m/2, lda, Ah, m/2);

    dim3 grid1((m/2+31)/32, (n+31)/32);
    dim3 block1(32,32);
    s2h<<<grid1, block1>>>(m/2, n, B, ldb, Bh, m/2);


    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m/2, n, m/2,
        &snegone, Ah, CUDA_R_16F, m/2, Bh, CUDA_R_16F, m/2,
        &sone, B+m/2, CUDA_R_32F, ldb, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );


    gemmTime +=stopTimer();
    //printf("%lf\n",gemmTime);
    //printMatrixDeviceBlock("ta.csv", m/2, m/2, A+m/2*m+m/2, lda);
    //printMatrixDeviceBlock("tb.csv", m/2, n, B+m/2, ldb);
    trsm(handle, m/2, n, A+m/2*m+m/2, lda, B+m/2, ldb, hwork);
    //printf("1111111\n");
    //printMatrixDeviceBlock("tx.csv", m/2, n, B+m/2, ldb);
    
}

void later_rtrsm(int m, int n, float* A, int lda, float* B, int ldb, __half* hwork)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    //printMatrixDeviceBlock("A.csv", m, m, A, lda);
    //printMatrixDeviceBlock("B.csv", m, n, B, ldb);
    trsm(handle, m, n, A, lda, B, ldb, hwork);
    printf("Panel takes %lfms\n", panelTime);
    printf("Gemm takes %lfms\n", gemmTime);
    //printf("22222\n");
    //printMatrixDeviceBlock("X.csv", m, n, B, ldb);

    cublasDestroy(handle);

    return;
}