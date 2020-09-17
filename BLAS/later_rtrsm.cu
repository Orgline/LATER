#include "LATER.h"

#include <cuda_fp16.h>

#define BLOCKSIZE 256

float sone = 1.0;
float snegone = -1.0;
float szero = 0.0;

float panelTime = 0.0;
float gemmTime = 0.0;



void trsm_l_l_n(cublasHandle_t handle, int m, int n, float* A, int lda, float* B, int ldb, __half* hwork)
{
    //printf("m,n=%d,%d\n", m, n);
    if(m <= BLOCKSIZE)
    {
        //startTimer();
        cublasStrsm(handle,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            m, n, &sone,
            A, lda,
            B, ldb
        );
        float ms = stopTimer();
        //panelTime += ms;
        //printf("panel TFLOPS is %lf\n", 1.0*m*m*n/1e9/ms);
        //printf("%lf\n",panelTime);
        return;
    }
    trsm_l_l_n(handle, m/2, n, A, lda, B, ldb, hwork);
    
    __half *Ah = hwork;
    __half *Bh = hwork+m/2*m/2;

    dim3 grid((m/2+31)/32, (m/2+31)/32);
    dim3 block(32,32);
    s2h<<<grid, block>>>(m/2, m/2, A+m/2, lda, Ah, m/2);

    dim3 grid1((m/2+31)/32, (n+31)/32);
    dim3 block1(32,32);
    s2h<<<grid1, block1>>>(m/2, n, B, ldb, Bh, m/2);
    //startTimer();

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m/2, n, m/2,
        &snegone, Ah, CUDA_R_16F, m/2, Bh, CUDA_R_16F, m/2,
        &sone, B+m/2, CUDA_R_32F, ldb, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    //float ms = stopTimer();
    //gemmTime += ms;
    //printf("GEMM flops is %lf\n", 2.0*m/2.0*n*m/2.0/1e9/ms);
    //printf("%lf\n",gemmTime);
    //printMatrixDeviceBlock("ta.csv", m/2, m/2, A+m/2*m+m/2, lda);
    //printMatrixDeviceBlock("tb.csv", m/2, n, B+m/2, ldb);
    trsm_l_l_n(handle, m/2, n, A+m/2*lda+m/2, lda, B+m/2, ldb, hwork);
    //printf("1111111\n");
    //printMatrixDeviceBlock("tx.csv", m/2, n, B+m/2, ldb);
    
}

void trsm_l_r_t(cublasHandle_t handle, int m, int n, float* A, int lda, float* B, int ldb, __half* hwork)
{
    
    if(n <= BLOCKSIZE)
    {
        //startTimer();
        cublasStrsm(handle,
            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            m, n, &sone,
            A, lda,
            B, ldb
        );
        //float ms = stopTimer();
        return;
    }
    //printf("1\n");
    trsm_l_r_t(handle, m, n/2, A, lda, B, ldb, hwork);
    //printf("1\n");
    __half *Ah = hwork;
    __half *Bh = hwork+n/2*n/2;

    dim3 grid((n/2+31)/32, (n/2+31)/32);
    dim3 block(32,32);
    s2h<<<grid, block>>>(n/2, n/2, A+n/2, lda, Ah, n/2);

    dim3 grid1((m+31)/32, (n/2+31)/32);
    dim3 block1(32,32);
    s2h<<<grid1, block1>>>(m, n/2, B, ldb, Bh, m);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n/2, n/2,
        &snegone, Bh, CUDA_R_16F, m, Ah, CUDA_R_16F, n/2,
        &sone, B+n/2*ldb, CUDA_R_32F, ldb, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    trsm_l_r_t(handle, m, n/2, A+n/2*lda+n/2, lda, B+n/2*ldb, ldb, hwork);
}

void later_rtrsm(char uplo, char leri, char trans, int m, int n, float* A, int lda, float* B, int ldb, __half* hwork)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    //printMatrixDeviceBlock("A.csv", m, m, A, lda);
    //printMatrixDeviceBlock("B.csv", m, n, B, ldb);
    if(uplo == 'l' && leri == 'l' && trans == 'n')
        trsm_l_l_n(handle, m, n, A, lda, B, ldb, hwork);
    else if (uplo == 'l' && leri == 'r' && trans == 't')
        trsm_l_r_t(handle, m, n, A, lda, B, ldb, hwork);
    //printf("Panel takes %lfms\n", panelTime);
    //printf("Gemm takes %lfms\n", gemmTime);
    //printf("22222\n");
    //printMatrixDeviceBlock("X.csv", m, n, B, ldb);

    cublasDestroy(handle);

    return;
}

/*
void later_rtrsm(int m, int n, float* A, int lda, float* B, int ldb, __half* hwork)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int i = 0; i<m; i+=BLOCKSIZE)
    {
        int nb = min(m-i, BLOCKSIZE);

        startTimer();
        //leaf op
        cublasStrsm(handle,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            nb, n, &sone,
            A+i, lda,
            B+i, ldb
        );

        panelTime+=stopTimer();

        startTimer();

        //if not the last block, then update
        if(m-i>BLOCKSIZE)
        {
            __half *Ah = hwork;
            __half *Bh = hwork + (m - i - nb)*nb;

            dim3 grid((m - i - nb + 31)/32, (nb+31)/32);
            dim3 block(32,32);
            s2h<<<grid, block>>>(m - i - nb, nb, A+i+nb, lda, Ah, m - i - nb);

            dim3 grid1((nb + 31)/32, (n+31)/32);
            dim3 block1(32,32);
            s2h<<<grid1, block1>>>(nb, n, B+i, ldb, Bh, nb);

            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m - i - nb, n, nb,
                &snegone, Ah, CUDA_R_16F, m - i - nb, Bh, CUDA_R_16F, nb,
                &sone, B+i+nb, CUDA_R_32F, ldb, CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
        }
        gemmTime+=stopTimer();
    }

    printf("Panel takes %lfms\n", panelTime);
    printf("Gemm takes %lfms\n", gemmTime);
}*/
