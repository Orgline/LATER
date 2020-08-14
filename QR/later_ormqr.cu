#include "LATER.h"
#include "LATER_QR.h"

/*
This routine aims to form the explict Q from WY representation

For the result from rhouqr routine:
The input will be 
W = [W1 W2]
Y is the Householder vectors
work is the workspace;
This routine will perform W = [W1 -W1*Y1'*W2+W2] at first;
Then it will form the explict Q=I-W*Y';
The output will be
W: the explicit Q

*/
void later_ormqr(int m, int n, float* W, int ldw, float* Y, int ldy, float *work)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    float sone = 1.0;
    float snegone = -1.0;
    float szero = 0.0;

    cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n/2,n/2,m,
        &sone,
        Y, ldy,
        W+ldw/2*n,ldw,
        &szero,
        work,n/2
    );

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m,n/2,n/2,
        &snegone,
        W, ldw,
        work,n/2,
        &sone,
        W+ldw/2*n,ldw
    );

    float *WI;
    cudaMalloc(&WI, sizeof(float)*m*n);
    dim3 grid1( (m+1)/32, (n+1)/32 );
	dim3 block1( 32, 32 );
    setEye<<<grid1,block1>>>( m, n, WI, m);


    cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,m,n,n,
        &snegone,W,CUDA_R_32F, ldw, Y, CUDA_R_32F, ldy,
        &sone, WI, CUDA_R_32F, m, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    );

    cudaMemcpy(W, WI, sizeof(float)*m*n, cudaMemcpyDeviceToDevice);

    cudaFree(WI);
    cublasDestroy(handle);
}

void later_ormqr2(int m, int n, float* W, int ldw, float* Y, int ldy, float* work)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    float sone = 1.0;
    float snegone = -1.0;

    dim3 grid1( (m+1)/32, (n+1)/32 );
	dim3 block1( 32, 32 );
    setEye<<<grid1,block1>>>( m, n, work, m);

    cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,m,n,n,
        &snegone,W,CUDA_R_32F, ldw, Y, CUDA_R_32F, ldy,
        &sone, work, CUDA_R_32F, m, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    );

    cudaMemcpy(W, work, sizeof(float)*m*n, cudaMemcpyDeviceToDevice);
}

