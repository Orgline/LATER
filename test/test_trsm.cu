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
    cudaMalloc(&A, sizeof(float)*m*m);
    float *B;
    cudaMalloc(&B, sizeof(float)*m*n);

    __half *hwork;
    cudaMalloc(&hwork, sizeof(__half)*(m/2*m/2+m/2*n));

    //generateUniformMatrix(A,m,m);
    float *hA;
    hA = (float*)malloc(sizeof(float)*m*m);
    for(long i=0;i<m*m;i++)
    {
        hA[i] = 0.1;
    }
    cudaMemcpy(A, hA, sizeof(float)*m*m, cudaMemcpyHostToDevice);
    dim3 grid((m+31)/32, (m+31)/32);
    dim3 block(32,32);
    clearTri<<<grid, block>>>('u', m, m, A, m);
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
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
        &sone, A, m, B, m,
        &szero, work, m
    );
    cudaMemcpy(B, work, sizeof(float)*m*n, cudaMemcpyDeviceToDevice);

    //generateNormalMatrix(B,m,n);
    //printMatrixDeviceBlock("A.csv", m, m, A, m);
    //printMatrixDeviceBlock("B.csv", m, n, B, m);
    later_rtrsm('l','l','n',m, n, A, m, B, m, hwork);
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
            &sone, A, m, dB, m,
            &szero, work, m
        );
        cudaMemcpy(dB, work, sizeof(float)*m*n, cudaMemcpyDeviceToDevice);

        float normB = snorm(m, n, dB);
        //printf("%lf\n", normB);

        
        //printMatrixDeviceBlock("dA.csv", m, m, A, m);
        //printMatrixDeviceBlock("dB.csv", m, n, dB, m);
        //printMatrixDeviceBlock("dX.csv", m, n, B, m);
        
        cublasSgemm(handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            m, n, m,
            &snegone, A, m,
            B, m,
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