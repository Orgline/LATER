#include "LATER.h"

#include <stdlib.h>

long m,n;
bool checkFlag;

int parseArguments(int argc,char *argv[])
{
    m = atoi(argv[1]);
    n = atoi(argv[2]);
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
    cudaMalloc(&A, sizeof(float)*n*n);
    float *B;
    cudaMalloc(&B, sizeof(float)*m*n);

    __half *hwork;
    cudaMalloc(&hwork, sizeof(__half)*(n/2*n/2+m/2*n));

    //generateUniformMatrix(A,m,m);
    float *hA;
    hA = (float*)malloc(sizeof(float)*n*n);
    for(long i=0;i<n*n;i++)
    {
        hA[i] = 0.1;
    }
    cudaMemcpy(A, hA, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    dim3 grid((n+31)/32, (n+31)/32);
    dim3 block(32,32);
    clearTri<<<grid, block>>>('u', n, n, A, n);
    //printf("debug 1\n");
    float *hB;
    hB= (float*)malloc(sizeof(float)*m*n);

    for(long i=0;i<m*n;i++)
        //hB[i] = rand()/(RAND_MAX+1.0)/m;
        hB[i] = 1.0;

    cudaMemcpy(B, hB, sizeof(float)*m*n, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    //printf("debug 1\n");
    float sone = 1.0;
    float snegone = -1.0;
    float szero = 0.0;

    float *work;
    cudaMalloc(&work, sizeof(float)*m*n);
    //printf("debug 1\n");
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
        &sone, B, m, A, n,
        &szero, work, m
    );
    cudaMemcpy(B, work, sizeof(float)*m*n, cudaMemcpyDeviceToDevice);

    //generateNormalMatrix(B,m,n);
    //printMatrixDeviceBlock("A.csv", m, m, A, m);
    //printMatrixDeviceBlock("B.csv", m, n, B, m);
    startTimer();

    later_rtrsm(handle, 'l','r','t',m, n, A, n, B, m, hwork);

    float ms = stopTimer();
    printf("rtrsm takes %f ms, flops is %f\n", ms, 1.0*m*n*n/ms/1e9);
    //printMatrixDeviceBlock("X.csv", m, n, B, m);
    //printf("debug 1\n");
    if(checkFlag)
    {
        //printf("Check backward error\n");
        float *dB;
        cudaMalloc(&dB, sizeof(float)*m*n);
        
        //generateNormalMatrix(dB,m,n);
        cudaMemcpy(dB, hB, sizeof(float)*m*n, cudaMemcpyHostToDevice);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
            &sone, dB, m, A, n,
            &szero, work, m
        );
        cudaMemcpy(dB, work, sizeof(float)*m*n, cudaMemcpyDeviceToDevice);

        float normB = snorm(m, n, dB);
        //printf("%lf\n", normB);

        
        //printMatrixDeviceBlock("dA.csv", m, m, A, m);
        //printMatrixDeviceBlock("dB.csv", m, n, dB, m);
        //printMatrixDeviceBlock("dX.csv", m, n, B, m);
        
        cublasSgemm(handle, 
            CUBLAS_OP_N, CUBLAS_OP_T, 
            m, n, n,
            &snegone, B, m,
            A, n,
            &sone, dB, m
        );
        //printMatrixDeviceBlock("bb.csv", m, n, dB, m);
        //printf("norm db = %.6e\n", snorm(m,n,dB));
        //cudaFree(dB);
        printf("Backward error ||A*X-B||/||B|| is %.6e\n", snorm(m,n,dB)/normB);
        cudaFree(dB);
    }

    {
        startTimer();
        cublasStrsm(handle,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            m, n, &sone,
            A, m,
            B, m
        );
        printf("cuSOLVER strsm takes %lf\n", stopTimer());
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(work);
    cudaFree(hwork);
    cublasDestroy(handle);
    
    return 0;
}