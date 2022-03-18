#include "LATER.h"
#include "LATER_QR.h"
#include <assert.h>
#include <string>

int n;
int ns;
bool checkFlag = false;

int parseArguments(int argc,char *argv[])
{
    n = atoi(argv[1]);
    ns = atoi(argv[2]);
    for (int i=3; i<argc; i++) {
        if(strcmp(argv[i], "-check") == 0) {
            checkFlag = true;
        }
    }
    return 0;
}

__global__
void generateSyMatrix(int m, int n, float* dA,int lda, float *tmpA)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		tmpA[i+j*lda] = (dA[i+j*lda] + dA[j+i*lda])/2.0f;
	}
    __syncthreads();
    if (i<m && j<n) {
        dA[i+j*lda] = tmpA[i+j*lda];
    }
}


int main(int argc,char *argv[])
{
    if(parseArguments(argc,argv)!=0)
    {
        return 0;
    }
    print_env();

    float *A;
    cudaMalloc(&A,sizeof(float)*n*n*2);
    float *H;
    cudaMalloc(&H, sizeof(float)*n*n);
    generateUniformMatrix(H, n ,n);
    float *tmpA;
    cudaMalloc(&tmpA, sizeof(float)*n*n);

    dim3 gridDim((n+31)/32,(n+31)/32);
    dim3 blockDim(32,32);
    generateSyMatrix<<<gridDim, blockDim>>>(n,n,H,n,tmpA);

    //printMatrixDeviceBlock("A.csv", n,n,H,n);

    deviceCopy<<<gridDim, blockDim>>>(n, n, H, n, tmpA, n);

    cudaCtxt ctxt {};
    cublasCreate(&ctxt.cublas_handle );
    cusolverDnCreate(&ctxt.cusolver_handle );
    int lda = 2*n;
    int ldh = n;

    float *work;
    cudaMalloc(&work, sizeof(float)*n*n);

    __half *hwork;
    cudaMalloc(&hwork, sizeof(__half)*2*n*n);

    //later_qdwh_polar(ctxt, n, A, lda, H, ldh, tmpA, work, hwork);
    return 0;
}