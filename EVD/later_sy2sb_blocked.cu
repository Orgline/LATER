#include <cuda_fp16.h>
#include <assert.h>
// #include <stdlib.h>
// #include <stdio.h>
#include <iostream>
#include <string.h>
#include "LATER.h"

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
	char name[] = "W0.csv";

	for(int i=0; i<(n-nb); i+=nb){
		int lm=n-i-nb;
		int ln=nb;
		printf("Itertaion %d :: Matrix size is %d*%d\n", i, lm, ln);
		name[0] = 'P';
		name[1] = (i/nb)+'0';
		printMatrixDeviceBlock(name, lm, ln,  &A[(i+nb)+i*lda], lda);
		CHECK_KERNEL();
		startTimer();
		later_rhouqr(lm, ln, &A[(i+nb)+i*lda], lda, &work[(i+nb)+i*n], n, &work[nb*n+i*nb], nb, &work[nb*n+nb*nb], lwork, hwork, lhwork, &work[lwork+lwork+nb*n+nb*nb]);
		// later_rhouqr(lm, ln, &A[(i+nb)+i*lda], lda, &W[(i+nb)+i*ldw], ldw, R, ldr, work, lwork, hwork, lhwork, U);
		float ms=stopTimer();
		float flops=2.0*lm*ln*lm;
		qr+=ms;
		printf("QR takes %fms, rate is %f TFLOPs\n", ms, flops/ms/1e9);
		name[0] = 'W';
		printMatrixDeviceBlock(name, lm, ln, &work[(i+nb)+i*n], n);
		name[0] = 'R';
		printMatrixDeviceBlock(name, ln, ln, &work[nb*n+i*nb], nb);
		name[0] = 'Y';
		printMatrixDeviceBlock(name, lm, ln, &A[(i+nb)+i*lda], lda);
		
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
		s2h<<<grid1,block1>>>(lm,lm,&A_cpy[(i+nb)+(i+nb)*lda],lda,buff1,lm);
		s2h<<<grid2,block1>>>(lm,ln,&work[(i+nb)+i*n],n,buff2,lm);
		auto status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, lm, ln, lm,
		&sones, buff1, CUDA_R_16F, lm, buff2, CUDA_R_16F, 
		lm, &szeros, buff_res, CUDA_R_16F, lm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		// assert(status == CUBLAS_STATUS_SUCCESS);
		ms=stopTimer();	
		flops=2.0*lm*ln*lm;
		p1+=ms;
		// printf("panel 1 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", lm, ln, lm, ms, flops/ms/1e9);

		//buff1 <- W'(AW) = W'(Z) = W'(buff_res)
		CHECK_KERNEL();
		startTimer();
		// s2h<<<gridDimA,blockDimA>>>(lm,ln,W,ldw,buff1,ldw);
		// s2h<<<gridDimA,blockDimA>>>(lm,ln,&Z[(i+nb)+i*ldz],ldz,&buff2[(i+nb)+i*ldz],ldz);		
		status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, ln, ln, lm, &sones, 
		buff2, CUDA_R_16F, lm, buff_res, CUDA_R_16F, lm, &szeros, 
		buff1, CUDA_R_16F, lm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		// assert(status == CUBLAS_STATUS_SUCCESS);
		ms=stopTimer(); 
		flops=2.0*ln*ln*lm;
		p2+=ms;
		// printf("panel 2 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", ln, ln, lm, ms, flops/ms/1e9);


		//Z <- Z - 1/2*Y*W
		//buff_res <- buff_res - 1/2*Y*W
		CHECK_KERNEL();
		startTimer();
		s2h<<<grid2,block1>>>(lm,ln,&A[(i+nb)+i*lda],lda,buff1,lm);
		// h2h<<<gridDimA,blockDimA>>>(ln,ln,&buff_res[i+i*lda],lda,&buff2[i+i*lda],lda);
		s2h<<<grid2,block1>>>(lm,ln,&work[(i+nb)+i*n],n,buff2,lm);
		status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, lm, ln, ln, &sneghalf,
		buff1, CUDA_R_16F, lm, buff2, CUDA_R_16F, lm, &sones,
		buff_res, CUDA_R_16F, lm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		// assert(status == CUBLAS_STATUS_SUCCESS);
		ms=stopTimer();
		p3+=ms;
		flops=2.0*lm*ln*ln;
		// printf("panel 3 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", lm, ln, ln, ms, flops/ms/1e9); 



		//A=Q'*A
		dim3 grid3((ln+31)/32,(ln+31)/32);
		s2s<<<grid3,block1>>>(ln,ln, &work[nb*n+i*nb],ln, &A_cpy[(i+nb)+i*lda],lda);
		
		
		// A=A'
		dim3 grid4((ln+31)/32,(lm+31)/32);
		copy_lower_to_upper<<<grid4,block1>>>(lm, ln, &A_cpy[i+(i+nb)*lda]);
		
		//A=A-YZ'
		CHECK_KERNEL();
		startTimer();
		s2h<<<grid2,block1>>>(lm,ln,&A[(i+nb)+i*lda],lda,buff1,lm);
		s2h<<<grid2,block1>>>(lm,ln,&A_cpy[(i+nb)+(i+nb)*lda],lda,buff2,lm);
		// s2h<<<grid2,block1>>>(lm,ln,&Z[(i+nb)+i*ldz],ldz,&buff2[(i+nb)+i*ldz],ldz);
		status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, lm, lm, ln, &snegones,
		buff1, CUDA_R_16F, lm, buff_res, CUDA_R_16F, lm, &sones,
		buff2, CUDA_R_16F, lm, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		// assert(status == CUBLAS_STATUS_SUCCESS);
		ms=stopTimer();
		p4+=ms;
		flops=2.0*lm*lm*ln;
		// printf("panel 4 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", lm, lm, ln, ms, flops/ms/1e9);

		//A=A-ZY'
		CHECK_KERNEL();
		startTimer();
		status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, lm, lm, ln, &snegones,
		buff_res, CUDA_R_16F, lm, buff1, CUDA_R_16F, lm, &sones,
		buff2, CUDA_R_16F, lda, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		// assert(status == CUBLAS_STATUS_SUCCESS);
		h2s<<<grid2,block1>>>(lm,ln,buff2,lm,&A_cpy[(i+nb)+(i+nb)*lda],lda);
		p5+=ms;
		ms=stopTimer();
		// printf("panel 5 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", lm, lm, ln, ms, flops/ms/1e9);
	}


	//printMatrixDeviceBlock("R.csv", nb, nb, R, ldr);
	printMatrixDeviceBlock("newA.csv", n, n, A, lda);

	printf("n::%d \nnb::%d \n", n, nb);
	printf("Total QR::%f ms\n", qr);
	printf("Total GEMM::%f ms (Panel1::%f ms, Panel2::%f ms, Panel3::%f ms, Panel4::%f ms, Panel5::%f ms)\n", (p1+p2+p3+p4+p5), p1, p2, p3, p4, p5);
	printf("norm(newA)::%f \n",snorm(n,n,A));

}
