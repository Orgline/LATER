#include "LATER.h"
#include "LATER_QR.h"
#include <assert.h>
#include <string>


#define NMIN 128

bool checkFlag = false;
int n;
int ns;
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

__global__
void setZeroOfWork( int m, int n, float *a, int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
        a[i+j*lda] = 0;
	}
}



int main(int argc,char *argv[])
{
    if (argc < 3) {
        printf("Usage: test n [options]\n");
        printf("Options:\n\t-check: enable checking the orthogonality and backward error\n");
        return 0;
    }
    if(parseArguments(argc,argv)!=0)
    {
        return 0;
    }
    print_env();

    int lda = n;

    //int ns = 512;;
    int lwork, lhwork;

    float *A;
    cudaMalloc(&A,sizeof(float)*n*n);
    float *oriA;
    cudaMalloc(&oriA,sizeof(float)*n*n);
    float *work;
    cudaMalloc(&work,sizeof(float)*n*n*2);
    __half *hwork;
    cudaMalloc(&hwork,sizeof(__half)*n*n*2);
    
    lwork = n*NMIN;
    lhwork = n*NMIN;
    cudaCtxt ctxt {};
    cublasCreate(&ctxt.cublas_handle );
    cusolverDnCreate(&ctxt.cusolver_handle );
    
    dim3 gridDim((n+31)/32,(n+31)/32);
    dim3 blockDim(32,32);
    generateNormalMatrix(A, n ,n);
    generateSyMatrix<<<gridDim, blockDim>>>(n,n,A,n,work);

    dim3 gridDimA((n*2+31)/32,(n+31)/32);
    setZeroOfWork<<<gridDimA, blockDim>>>(2*n, n, work, 2*n);
    cudaMemcpy(oriA, A, sizeof(float)*n*n, cudaMemcpyDeviceToDevice);
    later_rhouqr(ctxt, n, NMIN, A, lda, work, lda, work, NMIN, work, lwork, hwork, lhwork, work);
    

    

    
    // cudaEvent_t abegin, aend;
    // cudaEventCreate(&abegin);
    // cudaEventRecord(abegin);
    // cudaEventCreate(&aend);
    //startTimer();
    
        startTimer();
        //printMatrixDeviceBlock("A.csv", n-i-NMIN, NMIN, A + i+NMIN+i*lda, lda);
    generateUniformMatrix(A, n ,n);
    generateSyMatrix<<<gridDim, blockDim>>>(n,n,A,n,work);
    setZeroOfWork<<<gridDimA, blockDim>>>(2*n, n, work, 2*n);
    cudaMemcpy(oriA, A, sizeof(float)*n*n, cudaMemcpyDeviceToDevice);
    //printMatrixDeviceBlock("A.csv", n,n,A,n);
    later_sy2sb_rec(ctxt, n, ns, A, oriA, lda, work, lwork, hwork, lhwork);
    //printMatrixDeviceBlock("A.csv", n,n,A,n);
    // cudaEventRecord(aend);
    // cudaEventSynchronize(aend);
    // float milliseconds;
    // cudaEventElapsedTime(&milliseconds, abegin, aend);
    // cudaEventDestroy(abegin);
    // cudaEventDestroy(aend);
    //printf("takes %f ms\n", stopTimer());
    cudaFree(A);
    cudaFree(oriA);
    cudaFree(work);
    cudaFree(hwork);
}