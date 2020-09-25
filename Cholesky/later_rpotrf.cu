#include "LATER.h"

#include <cuda_fp16.h>

#include <assert.h>

#define BLOCKSIZE 2048
#define LWORK 65536

int chol_info;
int lwork;

int *dev_info;

float chol_panel = 0.0;
float chol_gemm = 0.0;

/*
This function performs recursive Cholesky factorization
*/

void l_potrf(cudaCtxt ctxt, int n, float* A, int lda, float* work, __half* hwork)
{
    float ms;
    //printf("n = %d\n", n);
    if(n<=BLOCKSIZE)
    {
        //printMatrixDeviceBlock("AAA.csv", n,n, A,lda);

        startTimer();
        cusolverDnSpotrf(ctxt.cusolver_handle ,
            CUBLAS_FILL_MODE_LOWER,
            n, A, lda,
            work, LWORK,
            dev_info);
        
        chol_panel += stopTimer();

        printf("panel takes %f ms\n", chol_panel);

        //assert(CUSOLVER_STATUS_SUCCESS == status);
        //printf("status = %d\n", status);
        //printMatrixDeviceBlock("LLL.csv", n,n, A,lda);
        
        //cudaMemcpy(&chol_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
        
        //printf("info = %d\n", chol_info);
        return;
    }

    l_potrf(ctxt, n/2, A, n, work, hwork);

    startTimer();

    later_rtrsm(ctxt.cublas_handle, 'l', 'r', 't', n/2, n/2, A, n, A+n/2, n, hwork);

    later_rsyrk(ctxt.cublas_handle, n/2, n/2, -1.0, A+n/2, n, 1.0, A+n/2*n+n/2, n, hwork);

    ms = stopTimer();
    
    chol_gemm+=ms;

    printf("n = %d,gemm takes %f ms, update flops is %f TFLOPS\n", n/2, chol_gemm, 2.0*n/2*n/2*n/2/ms/1e9);

    l_potrf(ctxt, n/2, A+n/2*n+n/2, n, work, hwork);
}

void later_rpotrf(char uplo, int n, float* A, int lda, float* work, __half* hwork)
{
    cudaCtxt ctxt;
    cublasCreate(&ctxt.cublas_handle);
    cusolverDnCreate(&ctxt.cusolver_handle);
    //printMatrixDeviceBlock("A.csv", n,n, A,n);
    cudaMalloc(&dev_info, sizeof(int));

    if(uplo == 'l')
    {
        
        l_potrf(ctxt, n, A, lda, work, hwork);
    }

    printf("Panel takes %f ms, update takes %f ms\n", chol_panel, chol_gemm);

    printf("TFLOPS is %lf TFLOPS\n", 1.0/3.0*n*n*n/(chol_gemm+chol_panel)/1e9);

    cublasDestroy(ctxt.cublas_handle);
    cusolverDnDestroy(ctxt.cusolver_handle);

    //printMatrixDeviceBlock("L.csv", n, n, A, n);

    return; 
}