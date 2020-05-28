#include "LATER.h"
#include "LATER_QR.h"

#include <cuda_fp16.h>

#define NMIN 32

/*
This routine performs recursive Householder QR factorization

The input A stores the original matrix A to be factorized
The output A stores the Householder vectors Y
The output W stores the W matrix of WY representation
The orthogonal matrix Q 
THe output R stor
*/

void qr(cudaCtxt ctxt, int m, int n, float *A, int lda, float *W, int ldw, float *R, int ldr, float *work, int lwork,
    __half *hwork, int lhwork, float* U);

void later_rhouqr(int m, int n, float* A, int lda, float* W, int ldw, float* R, int ldr, float* work, int lwork, __half* hwork, int lhwork, float* U)
{
    printf("Function rhouqr\n");
    
    cudaCtxt ctxt;
    cublasCreate( & ctxt.cublas_handle );
    cusolverDnCreate( & ctxt.cusolver_handle );

    qr(ctxt, m, n, A, lda, W, ldw, R, ldr, work, lwork, hwork, lhwork, U);
    
    cublasDestroy(ctxt.cublas_handle);
    cusolverDnDestroy(ctxt.cusolver_handle);
    return;
}

void qr(cudaCtxt ctxt, int m, int n, float *A, int lda, float *W, int ldw, float *R, int ldr, float *work, int lwork, __half *hwork, int lhwork, float* U)
{
    if(n<=NMIN)
    {
        return;
    }
}