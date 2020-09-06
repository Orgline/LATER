#include "LATER.h"

#include <cuda_fp16.h>

#define BLOCKSIZE 256


float sone1 = 1.0;
float snegone1 = 1.0;
float szero1 = 0.0;
float panelTime1 = 0.0;
float gemmTime1 = 0.0;

void trmm(cublasHandle_t handle, int m, int n, float* A, int lda, float* B, int ldb, float *C, int ldc, float *tempC,  __half* hwork)
{

        if(m <= BLOCKSIZE)
        {
                startTimer();
                cublasStrmm(handle,
                                CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                m, n,
                                &sone1,
                                A, lda,
                                B, ldb,
                                C, ldc);
                panelTime1 += stopTimer();
                return;
        }

        trmm(handle, m/2, n, A, lda, B, ldb, C, ldc, tempC, hwork);
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
                        &sone1, Ah, CUDA_R_16F, m/2, Bh, CUDA_R_16F, m/2,
                        &szero1, tempC, CUDA_R_32F, lda, CUDA_R_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP
                    );

        gemmTime1 +=stopTimer();

        startTimer();
        tempC = tempC+m/2 ;
        gemmTime1 +=stopTimer();

        trmm(handle, m/2, n, A+m/2*lda+m/2, lda, B+m/2, ldb, C+m/2, ldc, tempC, hwork);
        startTimer();
        tempC = tempC - m/2 ;

        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m/2, n,
                        &sone1,
                        tempC, lda,
                        &sone1,
                        C+m/2, ldc,
                        C+m/2, ldc);
        gemmTime1 +=stopTimer();
}


void later_rtrmm(int m, int n, float* A, int lda, float* B, int ldb, float *C, int ldc, float *tempC,  __half* hwork)
{
        cublasHandle_t handle;
        cublasCreate(&handle);


        trmm(handle, m, n, A, lda, B, ldb, C, ldc, tempC, hwork);
        printf("Panel takes %lfms\n", panelTime1);
        printf("Gemm takes %lfms\n", gemmTime1);

        cublasDestroy(handle);

        return;
}
