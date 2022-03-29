#include <cuda_fp16.h>
#include <assert.h>
// #include <stdlib.h>
// #include <stdio.h>
#include <iostream>
#include <string.h>
#include "LATER.h"

__global__
void copyRtoPanel_blocked(int m, int n, float* dA,int lda, float *dR, int ldr)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) 
    {
        if(i < j)
        	dA[i+j*lda] = dR[i+j*ldr];
		else
			dA[i+j*lda] = 0;       
	}
}

void ssytrd_sy2sb(cudaCtxt ctxt, int n, int nb, float *A, float* A_cpy, int lda, float* work, int lwork, __half* hwork, int lhwork){
	float qr=0.0;
	float p1=0.0;
	float p2=0.0;
	float p3=0.0;
	float p4=0.0;
	float p5=0.0;
	float sones = 1.0;
	float szeros = 0.0;
	float snegones = -1.0;
	float sneghalf= -0.5;

	float* W;
	cudaMalloc(&W,sizeof(float)*n*nb);
	float* R;
	cudaMalloc(&R,sizeof(float)*n*n);


	char name[] = "W0.csv";
	// printf("norm(origA)::%f \n",snorm(n,n,A_cpy));

	for(int i=0; i<(n-nb); i+=nb){
		int lm=n-i-nb;
		int ln=nb;
		// printf("Itertaion %d :: Matrix size is %d*%d\n", i, lm, ln);
		// name[0] = 'P';
		// name[1] = (i/nb)+'0';
		// printMatrixDeviceBlock(name, lm, ln,  &A[(i+nb)+i*lda], lda);
		CHECK_KERNEL();
		startTimer();
		later_rhouqr(ctxt, lm, ln, &A[(i+nb)+i*lda], lda, &W[(i+nb)+i*n], n, R, nb, work, lwork, hwork, lhwork, work+nb*n+nb*nb);
		float ms=stopTimer();
		float flops=2.0*lm*ln*lm;
		qr+=ms;
		printf("QR takes %fms, rate is %f TFLOPs\n", ms, flops/ms/1e9);
		// name[0] = 'W';
		// printMatrixDeviceBlock(name, lm, ln, &work[(i+nb)+i*n], n);
		// name[0] = 'R';
		// printMatrixDeviceBlock(name, ln, ln, &work[nb*n+i*nb], nb);
		// name[0] = 'Y';
		// printMatrixDeviceBlock(name, lm, ln, &A[(i+nb)+i*lda], lda);
		
		//Z = A*W - 1/2(Y*W'*A*W)
		__half *buff1 = hwork;
		__half *buff2 = hwork+(n-nb)*(n-nb);
		__half *buff_res = hwork+(n-nb)*(n-nb)+(n-nb)*nb;
		dim3 block1(32,32);
		dim3 grid1( (lm+31)/32,(lm+31)/32);
		dim3 grid2( (lm+31)/32,(ln+31)/32);


		//buff_res <- Z <- AW
		CHECK_KERNEL();
		startTimer();
		s2h<<<grid1,block1>>>(lm, lm, &A_cpy[(i+nb)+(i+nb)*lda], lda, buff2, lm);
		s2h<<<grid2,block1>>>(lm, ln, &W[(i+nb)+i*n], n, buff1, lm);
		auto status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, lm, ln, lm,
		&sones, buff2, CUDA_R_16F, lm, buff1, CUDA_R_16F, lm, &szeros, buff_res, 
		CUDA_R_16F, lm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		assert(status == CUBLAS_STATUS_SUCCESS);
		ms=stopTimer();	
		// __half* buff_aw = hwork;
		// h2h<<<grid2,block1>>>(lm, ln, buff_res, lm , buff_aw, lm);
		flops=2.0*lm*ln*lm;
		p1+=ms;
		printf("panel 1 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", lm, ln, lm, ms, flops/ms/1e9);
		// printMatrixDeviceBlock("A1.csv", lm, lm, &A_cpy[(i+nb)+(i+nb)*lda], lda);
		// printMatrixDeviceBlock("W1.csv", lm, ln, &W[(i+nb)+i*n], n);
		// float* AW;
		// cudaMalloc(&AW,sizeof(float)*lm*ln);
		// h2s<<<grid2,block1>>>(lm, ln, buff_res, lm , AW, lm);
		// printMatrixDeviceBlock("AW1.csv", lm, ln, AW, lm);



		//buff2 <- W'(AW) = W'(Z) = W'(buff_res)
		//buff1 is W and buff_res is Z
		CHECK_KERNEL();
		startTimer();
		s2h<<<grid2,block1>>>(lm, ln, &W[(i+nb)+i*n], n, buff1, lm);
		h2h<<<grid2,block1>>>(lm, ln, buff_res, lm, buff2, lm);
		cudaMemset(buff_res, 0, sizeof(__half)*ln*ln);
		status=cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, ln, ln, lm, &sones, 
		buff1, CUDA_R_16F, lm, buff2, CUDA_R_16F, lm,  &szeros, 
		buff_res, CUDA_R_16F, ln, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		assert(status == CUBLAS_STATUS_SUCCESS);
		ms=stopTimer();		
		flops=2.0*ln*ln*lm;
		p2+=ms;
		printf("panel 2 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", ln, ln, lm, ms, flops/ms/1e9);
		// float* WAW;
		// cudaMalloc(&WAW,sizeof(float)*ln*ln); 
		// dim3 grid3( (ln+31)/32,(ln+31)/32);
		// h2s<<<grid3,block1>>>(ln, ln, buff_res, ln, WAW, ln);
		// // printMatrixDeviceBlock("W1.csv", lm, ln, &W[(i+nb)+i*n], n);
		// // printMatrixDeviceBlock("AW1.csv", lm, ln, AW, lm);
		// printMatrixDeviceBlock("WAW1.csv", ln, ln, WAW, ln);

		//Z <- Z - 1/2*Y*(W'(AW)) = buff2 - 1/2*buff1*(buff_res)
		CHECK_KERNEL();
		startTimer();
		s2h<<<grid2,block1>>>(lm, ln, &A[(i+nb)+i*lda], lda, buff1, lm); 
		status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, lm, ln, ln, &sneghalf,
		buff1, CUDA_R_16F, lm, buff_res, CUDA_R_16F, ln, &sones,
		buff2, CUDA_R_16F, lm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		// assert(status == CUBLAS_STATUS_SUCCESS);
		ms=stopTimer();
		p3+=ms;
		flops=2.0*lm*ln*ln;
		printf("panel 3 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", lm, ln, ln, ms, flops/ms/1e9); 
		// float* Z;
		// cudaMalloc(&Z,sizeof(float)*lm*ln); 
		// h2s<<<grid2,block1>>>(lm, ln, buff2, lm, Z, lm);
		// printMatrixDeviceBlock("Y1.csv", lm, ln, &A[(i+nb)+i*lda], lda);
		// printMatrixDeviceBlock("Z1.csv", lm, ln, Z, lm);


		// // A=A'
		// copy_lower_to_upper<<<grid2,block1>>>(lm, lm, &A_cpy[i+(i+nb)*lda]);

		//A <- A-YZ' = A-(buff1)(buff2)'
		CHECK_KERNEL();
		startTimer();
		s2h<<<grid2,block1>>>(lm, ln, &A[(i+nb)+i*lda], lda, buff1, lm);
		// s2h<<<grid2,block1>>>(lm, ln, Z, lm, buff2, lm);
		status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, lm, lm, ln, &snegones,
		buff1, CUDA_R_16F, lm, buff2, CUDA_R_16F, lm, &sones,
		&A_cpy[(i+nb)+(i+nb)*lda], CUDA_R_32F, lda, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		// assert(status == CUBLAS_STATUS_SUCCESS);
		ms=stopTimer();
		p4+=ms;
		flops=2.0*lm*lm*ln;
		// printMatrixDeviceBlock("uA1.csv", lm, lm, &A_cpy[(i+nb)+(i+nb)*lda], lda);
		printf("panel 4 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", lm, lm, ln, ms, flops/ms/1e9);


		//A=A-ZY'
		CHECK_KERNEL();
		startTimer();
		s2h<<<grid2,block1>>>(lm, ln, &A[(i+nb)+i*lda], lda, buff1, lm);
		// s2h<<<grid2,block1>>>(lm, ln, Z, lm, buff2, lm);
		status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, lm, lm, ln, &snegones,
		buff2, CUDA_R_16F, lm, buff1, CUDA_R_16F, lm, &sones,
		&A_cpy[(i+nb)+(i+nb)*lda], CUDA_R_32F, lda, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		// assert(status == CUBLAS_STATUS_SUCCESS);
		ms=stopTimer();
		p5+=ms;
		// printMatrixDeviceBlock("uA2.csv", lm, lm, &A_cpy[(i+nb)+(i+nb)*lda], lda);
		printf("panel 5 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", lm, lm, ln, ms, flops/ms/1e9);

		//A=Q'*A
		copyRtoPanel_blocked<<<grid2,block1>>>(lm, lm, &A_cpy[(i+nb)+i*lda], lda, R, nb);
		// printMatrixDeviceBlock("R1.csv", ln, ln, R, nb);
		// printMatrixDeviceBlock("fA.csv", lm, lm, &A_cpy[(i+nb)+(i+nb)*lda], lda);

	}

	printMatrixDeviceBlock("A_block.csv", n, n, A_cpy, lda);

	printf("n::%d \nnb::%d \n", n, nb);
	printf("Total QR::%f ms\n", qr);
	printf("Total GEMM::%f ms (Panel1::%f ms, Panel2::%f ms, Panel3::%f ms, Panel4::%f ms, Panel5::%f ms)\n", (p1+p2+p3+p4+p5), p1, p2, p3, p4, p5);

}