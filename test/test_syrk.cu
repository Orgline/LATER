#include "LATER.h"

#include <stdlib.h>

long n, k;
bool checkFlag;

int parseArguments(int argc,char *argv[])
{
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    for (int i=3; i<argc; i++) {
        if(strcmp(argv[i], "-check") == 0) {
            checkFlag = true;
        }
    }
    return 0;
}

int main(int argc,char *argv[])
{
    if (argc < 3) {
        printf("Usage: test m n [options]\n");
        printf("Options:\n\t-check: enable checking the backward error\n");
        return 0;
    }
    if(parseArguments(argc,argv)!=0)
    {
        return 0;
    }
    float *A;
    cudaMalloc(&A, sizeof(float)*n*k);

    float *C;
    cudaMalloc(&C, sizeof(float)*n*n);

    __half *hwork;
    cudaMalloc(&hwork, sizeof(__half)*n*k);

    generateUniformMatrix(A,n,k);

    //startTimer();
    later_rsyrk(n, k, A, n, C, n, hwork);
    float ms;

    float *tC;
    cudaMalloc(&tC, sizeof(float)*n*n);

    cudaMemcpy(tC, C, sizeof(float)*n*n, cudaMemcpyDeviceToDevice);



    
    //printf("SYRK takes %lfms\n", ms);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float sone  = 1.0;
    float szero = 0.0;
    startTimer();
    
    cublasSsyrk(handle,
        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
        n, k,
        &sone,
        A, n,
        &szero,
        C, n
    );
    
    ms = stopTimer();

    printf("SYRK takes %lfms\n", ms);

    if(checkFlag)
    {
        float snegone = -1.0;
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
            &sone, C, n, &snegone, tC, n,
            C, n
        );
        printf("Forward error is %.6e\n",snorm(n,n,C)/snorm(n , n, tC));
    }

    startTimer();

    __half *Ah = hwork;

    dim3 grid((n+1)/32, (k+1)/32);
    dim3 block(32,32);
    s2h<<<grid, block>>>(n/2, k, A, n, Ah, n);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k,
        &sone, Ah, CUDA_R_16F, n, Ah, CUDA_R_16F, n,
        &szero, C, CUDA_R_32F, n, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    ms = stopTimer();

    printf("TC-GEMM takes %lf ms, %lf TFLOPS\n", ms, 2.0*n*n*k/1e9/ms);

    

    cublasDestroy(handle);

    cudaFree(A);
    cudaFree(hwork);
    cudaFree(C);
    cudaFree(tC);
}

