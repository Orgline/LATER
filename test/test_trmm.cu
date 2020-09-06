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

	float *hB;
	hB= (float*)malloc(sizeof(float)*m*n);

	for(long i=0;i<m*n;i++)
	{
		hB[i] = 1.0;
	}
	cudaMemcpy(B, hB, sizeof(float)*m*n, cudaMemcpyHostToDevice);

	float *C;
	cudaMalloc(&C, sizeof(float)*m*n);
	float *tempC = A; //cudaMalloc(&tempC, sizeof(float)* m *n);

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

	float *dC;
	if(checkFlag)
	{
		//printf("Check forwards error\n");
		//float *dC;
		cudaMalloc(&dC, sizeof(float)*m*n);

		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
				&sone, A, m, B, m,
				&szero, dC, m
			   );
	}


		later_rtrmm(m, n, A, m, B, m, C, m, tempC, hwork);


	if(checkFlag)
	{
		//printf("Check forwards error\n");
		//float *dC;
		/*	cudaMalloc(&dC, sizeof(float)*m*n);

			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
			&sone, A, m, B, m,
			&szero, dC, m
			); 
		 */
		float normC = snorm(m, n, dC);
		printf("normC = %lf\n", normC);


		cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
				&sone,
				dC, m,
				&snegone,
				C, m,
				dC, m);


		printf("Forward error ||C-hat(C)||/||C|| is %.6e\n", snorm(m,n,dC)/normC);

		cudaFree(dC);
	}


	{
		startTimer();
		cublasStrmm(handle,
				CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
				CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
				m, n,
				&sone,
				A, m,
				B, m,
				C, m);
		printf("cuSOLVER strmm takes %lf\n", stopTimer());
	}
	cudaFree(work);
	cudaFree(hwork);
/*	float ms;
	startTimer();
	__half *hAA;
	cudaMalloc(&hAA, sizeof(__half)*m*m);
	__half *hBB;*/
	//        cudaMalloc(&hBB, sizeof(__half)*m*n);

	//startTimer();

/*	dim3 grid1((m+1)/32, (n+1)/32);
	dim3 block1(32,32);
	s2h<<<grid, block>>>(m, m, A, m, hAA, m); // I am not doing*/ 
	cudaFree(A);

//	cudaMalloc(&hBB, sizeof(__half)*m*n);
//	s2h<<<grid1, block1>>>(m, n, B, n,hBB, n); // I am not doing 
	cudaFree(B);
/*	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, m,
			&sone, hAA, CUDA_R_16F, m, hBB, CUDA_R_16F, m,
			&szero, C, CUDA_R_32F, m, CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP
		    );*/
//	ms = stopTimer();

//	printf("NAVE TC-GEMM - TRMM takes %lf ms\n", ms);
	tempC = NULL;
	//	cudaFree(A);
	//cudaFree(B);
	cudaFree(C);    
	//    cudaFree(tempC);
	//	cudaFree(work);
	//	cudaFree(hwork);
	cublasDestroy(handle);

	return 0;
}
