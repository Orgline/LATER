#include "LATER.h"
#include "LATER_QR.h"

#include <cuda_fp16.h>

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

void qr(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work, int lwork, __half *hwork, int lhwork)
{

    if(n<=NMIN)
    {
        mgs_caqr_panel_256x128(ctxt,  m, n, A, lda, R, ldr, work );
        return;
    }

    //left recurse
    qr( ctxt, m, n/2, A, lda, R, ldr, work, lwork, hwork, lhwork );
    float sone = 1.0, szero = 0;
    float snegone= -1.0;

    __half *Ah = hwork;
    __half *Bh = &hwork[m*n/2];
    dim3 gridDim((m+31)/32,(n+31)/32);
    dim3 blockDim(32,32);
    s2h<<<gridDim, blockDim>>>(m, n/2, A, m, Ah, m);
    s2h<<<gridDim, blockDim>>>(m, n/2, &A[n/2*lda], m, Bh, m);
    cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n/2, n/2, m,
        &sone, Ah, CUDA_R_16F, lda, Bh, CUDA_R_16F, lda,
        &szero, &R[n/2*ldr], CUDA_R_32F, ldr, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    dim3 gridDim2( (n+31)/32, (n+31)/31 );
    s2h<<<gridDim2, blockDim>>>(n/2, n/2, &R[n/2*ldr], ldr, Bh, n/2);
    cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n/2, n/2,
        &snegone, Ah, CUDA_R_16F, m, Bh, CUDA_R_16F, n/2,
        &sone, &A[n/2*lda], CUDA_R_32F, lda, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    qr( ctxt, m, n/2, &A[n/2*lda], lda, &R[n/2+n/2*ldr], ldr, work, lwork, hwork, lhwork );

    return;
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

    //sqr()

    return;
}