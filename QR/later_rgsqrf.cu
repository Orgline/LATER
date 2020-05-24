#include "LATER.h"

/*
This function performs recursive Gram-Schmidt QR factorization

[A1|A2]=[Q1|Q2][R11|R12]
               [  0|R22]
A1=Q1*R11;
R12=Q1^T*A2;
A2=A2-Q1*R12;
A2=Q2*R22;

The input A stores the original matrix A to be factorized
the output A stores the orthogonal matrix Q
the output A stores the upper triangular matrix R

Both A and R need to be stored on GPU initially
*/

#define NMIN 128

int qr(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work, int lwork, __half *hwork, int lhwork)
{
    int info;

    if(n<=128)


    return 0;
}

void later_rgsqrf(int m, int n, float *A, int lda, float *R, int ldr)
{
    cudaCtxt ctxt;
    cublasCreate(&ctxt.cublas_handle );
    cusolverDnCreate(&ctxt.cusolver_handle );

    int lwork;

    cusolverDnSgeqrf_bufferSize(
        ctxt.cusolver_handle,
        m,
        NMIN,
        A,
        lda,
        &lwork
    );

    float *work;
    cudaMalloc( &work, lwork * sizeof(float) );

    __half *hwork;
	int lhwork = m*n;
    cudaMalloc( &hwork, sizeof(__half) * lhwork );

    qr()

    return;
}