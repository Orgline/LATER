#include "LATER.h"
#include "LATER_QR.h"
#include <math.h>

#include <cuda_fp16.h>

#define eps 2e-4

__global__
void generateNewU(int m, int n, float* dA,int lda)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		dA[i+j*lda] = (dA[i+j*lda] + dA[j+i*lda])/2.0f;
	}
}

void later_qdwh_polar()
{
    return;
}

// void later_qdwh_polar(cudaCtxt ctxt, int n, float *A, int lda, float *H, int ldh, float *tmpA, float *work, float *hwork)
// {
//     float alpha = 0.0;
//     cublasSnrm2(ctxt.cublas_handle, n*n, A, 1, &alpha);
//     alpha = 1.0/alpha;
//     cublasSscal(ctxt.cublas_handle, n*n, &alpha, A, 1);
//     float smin_est = 0.0;
//     float L = smin_est/sqrt(float(n));
//     float tol1 = 10*eps/2;
//     float tol3 = pow(tol1, 1.0/3);

//     dim3 gridDim((n+31)/32,(n+31)/32);
//     dim3 blockDim(32,32);
//     deviceCopy<<<gridDim, blockDim>>>(n, n, A, lda, tmpA, n);

//     setEye<<<gridDim, blockDim>>>(n, n, A+n, lda);

//     for(int iter = 0; iter < 10; iter++)
//     {
//         sSubstractAndSquare<<<gridDim, blockDim>>>(n, n, A, lda, tmpA,n);
//         float sum = 0.0;
//         cublasSasum(ctxt.cublas_handle, n*n, tmpA, 1, &sum);
//         sum = sqrt(sum);
//         if(sum < tol3 && iter > 0 && 1.0f-L < tol1)
//             break;
//         deviceCopy<<<gridDim, blockDim>>>(n, n, A, lda, work, n);
//         float L2 = L*L;
//         float dd = pow(4.0f*(1-L2)/(L2*L2), 1.0f/3.0f);
//         float sqd = sqrt(1+dd);
//         float a = sqd + sqrt(8 - 4*dd + 8*(2-L2)/(L2*sqd))/2;
//         float b = (a-1.0f)*(a-1.0f)/4.0f;
//         float c = a+b-1.0f;
//         L = L*(a+b*L2)/(1.0f+c*L2);;

//         int lwork = 2*n/256*32*n;
//         int lhwork = 2*n*n;

//         later_rgsqrf(ctxt, 2*n, n, A, lda, work, n, work, lwork, hwork, lhwork);
//         deviceCopy<<<gridDim, blockDim>>>(n, n, tmpA, n, work, n);
//         //alpha = b/c;
//         //cublasSscal(ctxt.cublas_handle, n*n, &alpha, work, 1);

//         __half *Q1 = hwork;
//         __half *Q2 = hwork+n*n;
//         s2h<<<gridDim, blockDim>>>(n, n, A, lda, Q1, n);
//         s2h<<<gridDim, blockDim>>>(n, n, A+n, lda, Q2, n);

//         alpha = (a-b/c)/sqrt(c);
//         float beta = b/c;

//         cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n,
//                      &alpha, Q1, CUDA_R_16F, n, Q2, CUDA_R_16F, n,
//                      &beta, &work, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
//                      CUBLAS_GEMM_DEFAULT_TENSOR_OP
//         );
//         generateNewU<<<gridDim, blockDim>>>(n, n ,work, n);
//         deviceCopy<<<gridDim, blockDim>>>(n, n, work, n, A, lda);
//         setEye<<<gridDim, blockDim>>>(n, n, A+n, lda);

//     }
//     return 0;
// }