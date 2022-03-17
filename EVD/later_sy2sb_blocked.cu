#include <cuda_fp16.h>
#include <assert.h>
#include "LATER.h"

void ssytrd_sy2sb(cudaCtxt ctxt, int n, int nb, float *A, float* A_cpy, int lda, float* U, int ldu, float* W, int ldw, float* R, int ldr, float* Z, int ldz, float* work, int lwork, __half* hwork, int lhwork){
	float qr=0.0;
	float p1=0.0;
        float p2=0.0;
	float p3=0.0;
	float p4=0.0;
	float p5=0.0;
	for(int i=0; i<(n-nb); i+=nb){
		int lm=n-i-nb;
		int ln=nb;
		startTimer();
		later_rhouqr(lm, ln, &A[(i+nb)+i*lda], lda, &W[(i+nb)+i*ldw], ldw, R, ldr, work, lwork, hwork, lhwork, U);
		float ms=stopTimer();
                float flops=2.0*lm*ln*lm;
                qr+=ms;
                printf("QR takes %fms, rate is %f TFLOPs\n", ms, flops/ms/1e9);
		//Z = A*W - 1/2(Y*W'*A*W)
		float sones = 1.0;
        	float szeros = 0.0;
		float snegones = -1.0;
		float sneghalf= -0.5;
		__half *buff1 = hwork;
        	__half *buff2 = hwork;
		__half *buff_res = hwork;
		dim3 gridDimA((n+31)/32,(n/2+31)/32);
        	dim3 blockDimA(32,32);

		//Z <- AW
       	 	s2h<<<gridDimA,blockDimA>>>(lm,lm,&A_cpy[(i+nb)+(i+nb)*lda],lda,&buff1[(i+nb)+(i+nb)*lda],lda);
                s2h<<<gridDimA,blockDimA>>>(lm,ln,&W[(i+nb)+i*ldw],ldw,&buff2[(i+nb)+i*ldw],ldw);
		startTimer();
		auto status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, lm, ln, lm,
                &sones, &buff1[(i+nb)+(i+nb)*lda], CUDA_R_16F, lda, &buff2[(i+nb)+i*ldw], CUDA_R_16F, 
		ldw, &szeros, &Z[(i+nb)+i*ldz], CUDA_R_32F, ldz, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                assert(status == CUBLAS_STATUS_SUCCESS);
       		ms=stopTimer();	
		flops=2.0*lm*ln*lm;
		p1+=ms;
		printf("panel 1 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", lm, ln, lm, ms, flops/ms/1e9);

		//buff_res <- W'(AW) = W'(Z)
		s2h<<<gridDimA,blockDimA>>>(lm,ln,&W[(i+nb)+i*ldw],ldw,&buff1[(i+nb)+i*ldw],ldw);
                s2h<<<gridDimA,blockDimA>>>(lm,ln,&Z[(i+nb)+i*ldz],ldz,&buff2[(i+nb)+i*ldz],ldz);		
		startTimer();
		status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, ln, ln, lm, &sones, 
		&buff1[(i+nb)+i*ldw], CUDA_R_16F, ldw, &buff2[(i+nb)+i*ldz], CUDA_R_16F, ldz, &szeros, 
		&buff_res[i+i*lda], CUDA_R_16F, ln, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
		assert(status == CUBLAS_STATUS_SUCCESS);
		ms=stopTimer(); 
		flops=2.0*ln*ln*lm;
		p2+=ms;
                printf("panel 2 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", ln, ln, lm, ms, flops/ms/1e9);


		//Z <- Z - 1/2*buff_res*W
		s2h<<<gridDimA,blockDimA>>>(lm,ln,&A[(i+nb)+i*lda],lda,&buff1[(i+nb)+i*lda],lda);
                h2h<<<gridDimA,blockDimA>>>(ln,ln,&buff_res[i+i*lda],lda,&buff2[i+i*lda],lda);
                startTimer();
		status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, lm, ln, ln, &sneghalf,
                &buff1[(i+nb)+i*lda], CUDA_R_16F, lda, &buff2[i+i*lda], CUDA_R_16F, ldw, &sones,
                &Z[(i+nb)+i*ldz], CUDA_R_32F, ldz, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
                assert(status == CUBLAS_STATUS_SUCCESS);
		ms=stopTimer();
		p3+=ms;
		flops=2.0*lm*ln*ln;
                printf("panel 3 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", lm, ln, ln, ms, flops/ms/1e9); 
       		
		//A=Q'*A
		s2s<<<gridDimA,blockDimA>>>(ln,ln,R,ldr,&A_cpy[(i+nb)+i*lda],lda);
		
		
		//A=A'
		float* tmpA = work;
		transpose<<<gridDimA,blockDimA>>>(ln, lm, &A_cpy[i+(i+nb)*lda], lda, tmpA);

		//A=A-YZ'
		s2h<<<gridDimA,blockDimA>>>(lm,ln,&A[(i+nb)+i*lda],lda,&buff1[(i+nb)+i*lda],lda);
                s2h<<<gridDimA,blockDimA>>>(lm,ln,&Z[(i+nb)+i*ldz],ldz,&buff2[(i+nb)+i*ldz],ldz);
                startTimer();
		status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, lm, lm, ln, &snegones,
                &buff1[(i+nb)+i*lda], CUDA_R_16F, lda, &buff2[(i+nb)+i*ldz], CUDA_R_16F, ldz, &sones,
                &A_cpy[(i+nb)+(i+nb)*lda], CUDA_R_32F, lda, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
                assert(status == CUBLAS_STATUS_SUCCESS);
		ms=stopTimer();
		p4+=ms;
                flops=2.0*lm*lm*ln;
                printf("panel 4 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", lm, lm, ln, ms, flops/ms/1e9);

		//A=A-ZY'
		startTimer();
                status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, lm, lm, ln, &snegones,
                &buff2[(i+nb)+i*ldz], CUDA_R_16F, ldz, &buff1[(i+nb)+i*lda], CUDA_R_16F, lda, &sones,
                &A_cpy[(i+nb)+(i+nb)*lda], CUDA_R_32F, lda, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
                assert(status == CUBLAS_STATUS_SUCCESS);
		p5+=ms;
		ms=stopTimer();
                printf("panel 5 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", lm, lm, ln, ms, flops/ms/1e9);
	}

	printf("n::%d \nnb::%d \n", n, nb);
	printf("Total QR::%f ms\n", qr);
	printf("Total GEMM::%f ms (Panel1::%f ms, Panel2::%f ms, Panel3::%f ms, Panel4::%f ms, Panel5::%f ms)\n", (p1+p2+p3+p4+p5), p1, p2, p3, p4, p5);
}
