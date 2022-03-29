#include "LATER.h"
#include <cuda_fp16.h>
// #include "OC_gemm.h"
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

	print_env();

	float *A;
	cudaMalloc(&A,sizeof(float)*n*n);
	int lda=n;    
	float *H;
	cudaMalloc(&H, sizeof(float)*n*n);
	generateUniformMatrix(A, n ,n);
	dim3 grid1((n+31)/32,(n+31)/32);
	dim3 block1(32,32);
	generateSyMatrix<<<grid1, block1>>>(n,n,A,lda,H);
	printMatrixDeviceBlock("A_orig_block.csv", n, n, A, lda);

	float* AA;
	cudaMalloc(&AA,sizeof(float)*n*n);
	cudaMemcpy(AA, A, sizeof(float)*n*n, cudaMemcpyDeviceToDevice); 

	int lhwork=n*nb;
	int lwork=n*nb;

   	float *work;
    cudaMalloc(&work,sizeof(float)*n*nb+nb*nb);
	cudaMemset(work, 0, sizeof(float)*n*nb+nb*nb);
    __half *hwork;
    cudaMalloc(&hwork,sizeof(__half)*n*nb+nb*nb);

	float* Dummy;
	cudaMalloc(&Dummy,sizeof(float)*n*n);
	cudaMemcpy(Dummy, A, sizeof(float)*n*n, cudaMemcpyDeviceToDevice); 
	later_rhouqr(ctxt, n, nb, Dummy, n, work, n, work, nb, work, lwork, hwork, lhwork, work);
	
	// startTimer();
	ssytrd_sy2sb(ctxt, n, nb, A, AA, lda, work, lwork, hwork, lhwork);
	// float ms=stopTimer();
	// printf("SY2SB takes %f ms\n", ms);

	cudaFree(A);
	cudaFree(H);
	cudaFree(hwork);
	cudaFree(work);
	cublasDestroy(ctxt.cublas_handle);
	cusolverDnDestroy(ctxt.cusolver_handle);
	return 0;
}
