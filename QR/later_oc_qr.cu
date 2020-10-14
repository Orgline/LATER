#include "LATER.h"

#define BLOCKSIZE 128

/*
Output:
A stores the Q on host
R stores the R on host
*/
void panel(int m, int n, float *A, int lda, float *R, int ldr, float* dA, float *dB, float *dR, float *hwork)
{
    cudaMemcpy(dA, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    int lwork = m / 256 * 32 * n;
    int lhwork = m * n;
    later_rgsqrf(m, n, dA, lda, dR, n, dB, lwork, hwork, lhwork);
    cublasGetMatrix(m, n, sizeof(float), dA, lda, A, lda);
    cublasGetMatrix(n, n, sizeof(float), dR, n, R, ldr);
}

void later_oc_qr(int m, int n, float *A, int lda, float *R, int ldr, float* dA, float *dB, float *dR, __half *hwork)
{
    if(n <= BLOCKSIZE)
    {
        panel(m, n, A, lda, R, ldr, dA, dB, dR, hwork);
        return;
    }
    //left recurse
    later_oc_qr(m, n / 2, A, lda, R, ldr, dA, dB, dR, hwork);

    float sone = 1.0;
    float szero = 0.0;
    float snegone = -1.0;

    later_oc_sgemm(CUBLAS_OP_T, CUBLAS_OP_N, n/2, n/2, m, &sone, A, lda, A+n/2*lda, lda, &szero, R+n/2*ldr, ldr);

    later_oc_sgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n/2, n/2, &snegone, A, lda, R+n/2*ldr, ldr, &sone, A+n/2*lda, lda);

    later_oc_qr(m, n/2, A+n/2*lda, lda, R+n/2+n/2*ldr, ldr, dA, dB, dR, hwork);
}

void later_oc_qr_blk(int m, int n, float *A, int lda, float *R, int ldr, float* dA, float *dB, float *dR, __half *hwork)
{
    for(int i = 0; i < n; i += BLOCKSIZE)
    {
        panel(m, n, A+i*lda, lda, R+i+i*ldr, ldr, dA, dB, dR, hwork);

        if(i < n - BLOCKSIZE)
        {
            later_oc_sgemm(CUBLAS_OP_T, CUBLAS_OP_N, BLOCKSIZE, n-i-BLOCKSIZE, m, &sone, A+i*lda, lda, A+(i+BLOCKSIZE)*lda, lda, &szero, R+i+BLOCKSIZE*ldr, ldr);

            later_oc_sgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n-i-BLOCKSIZE, BLOCKSIZE , &snegone, A+i*lda, lda, , &szero, R+i+BLOCKSIZE*ldr, ldr, A+(i+BLOCKSIZE)*lda, lda);
        }
    }
}

