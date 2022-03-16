#include "LATER.h"
#include "LATER_QR.h"
#include <math.h>

#include <cuda_fp16.h>

#define eps 2e-4

__global__
void generateNewU(int m, int n, float* dA,int lda, float *tmpA)
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


void later_qdwh_polar(cudaCtxt ctxt, int n, float *A, int lda, float *H, int ldh, float *tmpA, float *work, __half *hwork)
{
    float alpha = 0.0;
    cublasSnrm2(ctxt.cublas_handle, n*n, tmpA, 1, &alpha);
	//alpha = 63.9913;
    alpha = 1.0/alpha;

	printf("alpha = %lf\n", 1.0/alpha);
    cublasSscal(ctxt.cublas_handle, n*n, &alpha, tmpA, 1);

	//printMatrixDeviceBlock("A.csv", n,n,tmpA,n);
    //smin_est = Norm(tmpA, 1)/condest(tmpA)
    float smin_est = 0.0002070391384;
    float L = smin_est/sqrt(float(n));

    float tol1 = 10*eps/2;
    float tol3 = pow(tol1, 1.0/3);
    printf("smin_est = %.10f, sqrt(float(n) = %.10f, L = %.10f\n", smin_est, sqrt(float(n)), L);
    dim3 gridDim((n+31)/32,(n+31)/32);
    dim3 blockDim(32,32);
    deviceCopy<<<gridDim, blockDim>>>(n, n, tmpA, n, A, lda);

    

    for(int iter = 0; iter < 10; iter++)
    {
        startTimer();
        float sum = 0.0;
        if(iter > 0)
        {
            sSubstractAndSquare<<<gridDim, blockDim>>>(n, n, tmpA, n, work, n);
            cublasSasum(ctxt.cublas_handle, n*n, work, 1, &sum);
            sum = sqrt(sum);
        }
        if(sum < tol3 && iter > 0 && 1.0f-L < tol1)
            break;
        if(iter > 0)
            deviceCopy<<<gridDim, blockDim>>>(n, n, A, lda, tmpA, n);
        float L2 = L*L;
        float dd = pow(4.0f*(1-L2)/(L2*L2), 1.0f/3.0f);
        float sqd = sqrt(1+dd);
        float a = sqd + sqrt(8 - 4*dd + 8*(2-L2)/(L2*sqd))/2;
        float b = (a-1.0f)*(a-1.0f)/4.0f;
        float c = a+b-1.0f;
        L = L*(a+b*L2)/(1.0f+c*L2);

        printf("L,a,b,c=%f,%f,%f,%f\n", L, a, b, c);
        //deviceCopy<<<gridDim, blockDim>>>(n,n,work, n, A, lda);
        int lwork = 2*n/256*32*n;
        int lhwork = 2*n*n;
        float sqrtc = sqrt(c);
        cublasSscal(ctxt.cublas_handle, 2*n*n, &sqrtc, A, 1);
        
        setEye<<<gridDim, blockDim>>>(n, n, A+n, lda);
        
        later_rgsqrf(ctxt, 2*n, n, A, lda, work, n, work, lwork, hwork, lhwork);
        deviceCopy<<<gridDim, blockDim>>>(n, n, tmpA, n, work, n);
        //printMatrixDeviceBlock("A.csv", 2*n,n,A,lda);
        
        //alpha = b/c;
        //cublasSscal(ctxt.cublas_handle, n*n, &alpha, work, 1);

        __half *Q1 = hwork;
        __half *Q2 = hwork+n*n;
        s2h<<<gridDim, blockDim>>>(n, n, A, lda, Q1, n);
        s2h<<<gridDim, blockDim>>>(n, n, A+n, lda, Q2, n);
        
        alpha = (a-b/c)/sqrt(c);
        float beta = b/c;
        
        cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n,
                     &alpha, Q1, CUDA_R_16F, n, Q2, CUDA_R_16F, n,
                     &beta, work, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        
        generateNewU<<<gridDim, blockDim>>>(n, n ,work, n, A);
        
        deviceCopy<<<gridDim, blockDim>>>(n, n, work, n, A, lda);
        //setEye<<<gridDim, blockDim>>>(n, n, A+n, lda);
        float ms = stopTimer();
        printf("iteration takes %lf ms\n", ms);
    }
    //printMatrixDeviceBlock("U.csv", n,n,A,lda);

    return;
}

// float s_one = 1.0;
// float s_zero = 0.0;
// float s_negone = -1.0;
// int mmm = 32768;
// float rec_ms = 0.0;
// float rec_flops = 0.0;

// void recur_tri(cudaCtxt ctxt, int mm, int nn, __half *A, __half *B, float *C, int m, int n)
// {
// 	if(nn <= 128)
// 	{
// 		mmm -= 128;
// 		return;
// 	}
// 	recur_tri(ctxt, mm, nn/2, A, B, C, m, n);
// 	startTimer();
// 	cublasStatus_t t = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, mmm, nn/2, nn/2,
//         &s_negone, A, CUDA_R_16F, m, B, CUDA_R_16F, m,
//         &s_one, C, CUDA_R_16F, m, CUBLAS_COMPUTE_32F,
//         CUBLAS_GEMM_DEFAULT_TENSOR_OP
//     );
// 	float tmp = stopTimer();
// 	printf("GEMM size %dx%dx%d takes %.3f ms, exec rate %.3f TFLOPS\n", mmm, nn/2, nn/2,  tmp, 
//             2.0*(mmm)*(nn/2)*(nn/2)/(tmp*1e9));
// 	rec_ms += tmp;
// 	rec_flops += (2.0*(mmm)*(nn/2)*(nn/2));
// 	//printf("t = %d\n");
// 	recur_tri(ctxt,mm, nn/2, A, B, C, m, n);

// }

// void later_qdwh_polar()
// {
// 	cudaCtxt ctxt;
//     cublasCreate(&ctxt.cublas_handle );
//     cusolverDnCreate(&ctxt.cusolver_handle );
// 	int m = 32768, n = 32768, k = 128;
	
// 	__half *Ah;
// 	__half *Bh;
// 	float *C;
// 	cudaMalloc(&Ah, sizeof(__half)*m*n);
// 	cudaMalloc(&Bh, sizeof(__half)*n*n);
// 	cudaMalloc(&C, sizeof(float)*m*n);
// 	float ms = 0;
// 	float flops = 0.0;
// 	for(int i = m-128; i > 0; i = i-128)
// 	{
// 		startTimer();
// 		cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, i, i, k,
//         	&s_negone, Ah, CUDA_R_16F, m, Bh, CUDA_R_16F, m,
//         	&s_one, C, CUDA_R_16F, m, CUBLAS_COMPUTE_32F,
//         	CUBLAS_GEMM_DEFAULT_TENSOR_OP
//     	);
// 		float tmp = stopTimer();
// 		printf("GEMM size %dx%dx%d takes %.3f ms, exec rate %.3f TFLOPS\n", i, i, k,  tmp, 
//                 2.0*i*i*k/(tmp*1e9));
// 		flops = flops + 2.0*i*i*k;
// 		ms = tmp + ms;
// 	}
// 	printf("FLOP is %.0lf, Overall rate is %.3f TFLOPS\n",flops, flops/(ms)/1e9);


// 	recur_tri(ctxt, m, n, Ah, Bh, C, m, n);
// 	printf("FLOP is %.0lf, Overall rate is %.3f TFLOPS\n",rec_flops, rec_flops/(rec_ms)/1e9);
//     return;
// }

