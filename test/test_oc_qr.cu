#include "LATER.h"
#include "OC_gemm.h"

#define BLOCKSIZE 128
#define MEMORYSIZE 10240

int m,n;
int algo;

bool checkFlag = false;

int parseArguments(int argc,char *argv[])
{
    algo = atoi(argv[1]);
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    for (int i=4; i<argc; i++) {
        if(strcmp(argv[i], "-check") == 0) {
            checkFlag = true;
        }
    }
    return 0;
}

void generateRandomMatirx(int m, int n, float *A, int lda)
{
    for(long j = 0; j < n; j++)
    {
        for(long i = 0; i < m; i++)
        {
            A[i + j * lda] = rand()%1000/1000.0;
        }
    }
}

int main(int argc,char *argv[])
{
    if (argc < 4) {
        printf("Usage: test algo m n [options]\n");
        printf("Options:\n\t-check: enable checking the orthogonality and backward error\n");
        return 0;
    }
    if(parseArguments(argc,argv)!=0)
    {
        return 0;
    }
    print_env();

    float *A = (float*)malloc(sizeof(float)*m*n);
    float *R = (float*)malloc(sizeof(float)*n*n);

    generateRandomMatirx(m, n, A, m);

    long mem = MEMORYSIZE * (long)1024 * (long)1024;

    auto pool = std::make_shared<Mem_pool>(mem);

    float *work;
    __half *hwork;
    float *dA, *dR;
    /*
    cudaMalloc(&dA, sizeof(float)*m*BLOCKSIZE);
    cudaMalloc(&dR, sizeof(float)*BLOCKSIZE*BLOCKSIZE);
    cudaMalloc(&work, sizeof(float)*m/256*32*BLOCKSIZE);
    cudaMalloc(&hwork, sizeof(__half)*m*BLOCKSIZE);
    */
    
    cudaCtxt ctxt;
    cublasCreate(&ctxt.cublas_handle );
    cusolverDnCreate(&ctxt.cusolver_handle );

    if(algo == 1)
    {
        later_oc_qr_rec(ctxt, m, n, A, m, R, n, pool);
    }
    else if(algo == 2)
    {
        later_oc_qr_blk(ctxt, m, n, A, m, R, n, pool);
    }

    cublasDestroy(ctxt.cublas_handle);
    cusolverDnDestroy(ctxt.cusolver_handle);
    printf("Done\n");
}
