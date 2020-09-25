#include "LATER.h"

int n;

int algo;

bool checkFlag = false;

int parseArguments(int argc,char *argv[])
{
    algo = atoi(argv[1]);
    n = atoi(argv[2]);
    printf("algo = %d, n = %d\n", algo, n);
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
        printf("Usage: test algo m n [options]\n");
        printf("Options:\n\t-check: enable checking the orthogonality and backward error\n");
        return 0;
    }
    if(parseArguments(argc,argv)!=0)
    {
        return 0;
    }
    float *hA;
    hA = (float*)malloc(sizeof(float)*n*n);
    float *A;
    cudaMalloc(&A, sizeof(float)*n*n);

    for(long i = 0; i< n*n ;i++)
    {
        hA[i]=0.1;
    }

    cudaMemcpy(A, hA, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    dim3 grid((n+31)/32, (n+31)/32);
    dim3 block(32,32);
    clearTri<<<grid, block>>>('u', n, n, A, n);

    float *twork;
    cudaMalloc(&twork, sizeof(float)*n*n);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float sone = 1.0;
    float snegone = -1.0;
    float szero = 0.0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n,
        &sone, A, n, A, n,
        &szero, twork, n
    );

    cudaMemcpy(A, twork, sizeof(float)*n*n, cudaMemcpyDeviceToDevice);

    float normA = snorm(n,n,A);

    //cudaFree(twork);

    

    float *work;
    cudaMalloc(&work, sizeof(float)*128*128);

    __half *hwork;
    cudaMalloc(&hwork, sizeof(__half)*n/2*n);

    //printMatrixDeviceBlock("AA.csv", n,n, A,n);

    //printf("n = %d\n", n);

    later_rpotrf('l', n ,A, n, work, hwork);

    //printMatrixDeviceBlock("LL.csv", n,n, A,n);

    if(checkFlag)
    {
        clearTri<<<grid, block>>>('u', n, n, A, n);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n,
            &snegone, A, n, A, n,
            &sone, twork, n
        );

        printf("Backward error ||LL^T-A||/||A|| = %.6e\n", snorm(n, n, twork)/normA);

    }

    cudaFree(A);
    cudaFree(twork);
    cudaFree(work);
    cudaFree(hwork);

}