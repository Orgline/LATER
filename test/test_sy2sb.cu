#include "LATER.h"
//#include "LATER_QR.h"
#include <cuda_fp16.h>
#include "OC_gemm.h"
#include <random>
#include <assert.h>
#include <stdio.h>

__global__
void generateSyMatrix(int m, int n, float* dA,int lda, float *tmpA){
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


int main(int argc,char *argv[]){
    	if (argc<2) exit(1);

	int n = atoi(argv[1]);
    	int nb = atoi(argv[2]);

	cudaCtxt ctxt;
    	cublasCreate(&ctxt.cublas_handle );
    	cusolverDnCreate(&ctxt.cusolver_handle );

	float *A;
    	cudaMalloc(&A,sizeof(float)*n*n);
	int lda=n;    
	float *H;
	cudaMalloc(&H, sizeof(float)*n*n);
	generateUniformMatrix(A, n ,n);

	dim3 grid1((n+31)/32,(n+31)/32);
	dim3 block1(32,32);
	generateSyMatrix<<<grid1, block1>>>(n,n,A,lda,H);
//	printMatrixDeviceBlock("A.csv", n, n, A, lda);

	float* AA;
        cudaMalloc(&AA,sizeof(float)*n*n);
        cudaMemcpy(AA, A, sizeof(float)*n*n, cudaMemcpyDeviceToDevice); 
	__half *hwork;
        int lhwork = n*n;
        cudaMalloc( &hwork, sizeof(__half) * lhwork );
        float *U;
        cudaMalloc(&U,sizeof(float)*nb*nb);
        float *W;
        cudaMalloc(&W,sizeof(float)*n*n);
        float *R;
        cudaMalloc(&R,sizeof(float)*nb*nb);
	float *Z;
        cudaMalloc(&Z,sizeof(float)*n*n);
	float *work;
        int lwork = n/256*32*n;
        cudaMalloc(&work, sizeof(float)*lwork);
//	startTimer();
	ssytrd_sy2sb(ctxt, n, nb, A, AA, lda, U, nb, W, n, R, nb, Z, n, work, lwork, hwork, lhwork);
//	float ms=stopTimer();
//	printf("SY2SB takes %f ms\n", ms);

	cudaFree(A);
	cudaFree(H);
	cudaFree(hwork);
	cudaFree(U);
	cudaFree(W);
	cudaFree(R);
	cublasDestroy(ctxt.cublas_handle);
    	cusolverDnDestroy(ctxt.cusolver_handle);
	return 0;
}
