#include "LATER.h"

#include <cuda_fp16.h>

#include <assert.h>

#define BLOCKSIZE 128
#define LWORK 65536

int chol_info;
int lwork;

/*
This function performs recursive Cholesky factorization
*/

void l_potrf(cudaCtxt ctxt, int n, float* A, int lda, float* work, __half* hwork)
{
    printf("n = %d\n", n);
    if(n<=BLOCKSIZE)
    {
        cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
        cusolverDnSpotrf_bufferSize(ctxt.cusolver_handle,
            CUBLAS_FILL_MODE_LOWER,
            n,
            A,
            lda,
            &lwork );
        
        assert(CUSOLVER_STATUS_SUCCESS == status);
        
        printf("lwork = %d\n", lwork);

        printMatrixDeviceBlock("AAA.csv", n,n, A,lda);

        status = cusolverDnSpotrf(ctxt.cusolver_handle ,
            CUBLAS_FILL_MODE_LOWER,
            n, A, lda,
            work, LWORK,
            &chol_info);

        //assert(CUSOLVER_STATUS_SUCCESS == status);
        printf("status = %d\n", status);
        printMatrixDeviceBlock("LLL.csv", n,n, A,lda);
        
        printf("info = %d\n", chol_info);
        return;
    }

}

void later_rpotrf(char uplo, int n, float* A, int lda, float* work, __half* hwork)
{
    cudaCtxt ctxt;
    cublasCreate(&ctxt.cublas_handle);
    cusolverDnCreate(&ctxt.cusolver_handle);
    //printMatrixDeviceBlock("A.csv", n,n, A,n);

    if(uplo == 'l')
    {
        
        l_potrf(ctxt, n, A, lda, work, hwork);
    }

    cublasDestroy(ctxt.cublas_handle);
    cusolverDnDestroy(ctxt.cusolver_handle);

    //printMatrixDeviceBlock("L.csv", n, n, A, n);

    return; 
}