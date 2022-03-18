#include "LATER.h"
#include "LATER_QR.h"

#include <cuda_fp16.h>
#include <assert.h>

#define NMIN 128

__global__
void copyRtoPanel(int m, int n, float* dA,int lda, float *dR, int ldr)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
        if(j >= i)
		    dA[i+j*lda] = dR[i+j*ldr];
	}
}

float flops = 0.0;
float total_flops = 0.0;
float ms =0.0;
float total_ms = 0.0;
float total_zy = 0.0;


void later_sy2sb_rec(cudaCtxt ctxt, int n, int ns, float *A, float *oriA, int lda, float *work, int lwork, __half *hwork, int lhwork)
{
    printf("n = %d\n", n);
    float sone = 1.0f;
    float snegone = -1.0f;
    float szero = 0.0f;
    for(int i = 0; i < ns; i+=NMIN)
    {
        int lwork = n/256*32*NMIN;
        int lhwork = n*NMIN;
        //startTimer();
        later_rhouqr(n - i, NMIN, &A[i+NMIN+i*lda], lda, work+i+NMIN+i*n, n, work+ns*n, NMIN, work+ns*n+NMIN*NMIN, lwork, hwork, lhwork, work+ns*n+NMIN*NMIN);
        //ms = stopTimer();
        //total_ms += ms;

        if(i > 0)
        {
            /*form the new W matrix*/
            startTimer();
            __half *Ah = hwork;
            __half *Bh = hwork + i * (n - NMIN);
            __half *Ch = hwork + i * (n - NMIN) + NMIN * (n - NMIN);
            
            dim3 gridDimA((n-NMIN+31)/32,(i+31)/32);
            dim3 blockDimA(32,32);
            s2h<<<gridDimA,blockDimA>>>(n - NMIN, i, A + NMIN, lda , Ah , n - NMIN);
            dim3 gridDimB((n-NMIN+31)/32,(NMIN+31)/32);
            s2h<<<gridDimB,blockDimA>>>(n - NMIN, NMIN, work+ NMIN + i * n, n, Bh, n - NMIN);
            CHECK_KERNEL();
            
            auto status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, i, NMIN, n - NMIN,
                                       &sone, Ah, CUDA_R_16F, n - NMIN, Bh, CUDA_R_16F, n - NMIN,
                                       &szero, Ch, CUDA_R_16F, i, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                    );

            ms = stopTimer();
            flops = 2.0f*i*NMIN*(n-NMIN);
            total_ms = ms + total_ms;
            total_flops = flops + total_flops;
            printf("Form W 1 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", i, NMIN, n-NMIN, ms, 2.0f*i*NMIN*(n-NMIN)/ms/1e9);
            //assert(status == CUBLAS_STATUS_SUCCESS);
            CHECK_KERNEL();

            startTimer();
            s2h<<<gridDimA,blockDimA>>>(n - NMIN, i, work + NMIN, n, Ah, n - NMIN);
            CHECK_KERNEL();
            status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - NMIN, NMIN, i,
                                       &snegone, Ah, CUDA_R_16F, n - NMIN, Ch, CUDA_R_16F, i + NMIN,
                                       &sone, work+NMIN+i*n, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                    );
            ms = stopTimer();
            flops = 2.0f*i*NMIN*(n-NMIN);
            total_ms = ms + total_ms;
            total_flops = flops + total_flops;
            printf("Form w 2 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n - NMIN, NMIN, i, ms, flops/ms/1e9);

            //assert(status == CUBLAS_STATUS_SUCCESS);
            CHECK_KERNEL();
        }
        
        if(i + NMIN >= ns && n > ns)
        {
            /*if this is the last iteration, update the whole trailing matrix*/
            startTimer();
            __half *Ah = hwork;
            __half *Bh = hwork + (n - NMIN) * (n - NMIN);
            __half *Ch = hwork + (n - NMIN) * (n - NMIN) + (n - NMIN) * ns;
            dim3 gridDimA((n-NMIN+31)/32,(n-NMIN+31)/32);
            dim3 blockDimA(32,32);
            s2h<<<gridDimA,blockDimA>>>(n - NMIN, n - NMIN, oriA + NMIN, lda, Ah ,n-NMIN);
            dim3 gridDimB((n-NMIN+31)/32,(ns+31)/32);
            s2h<<<gridDimB,blockDimA>>>(n - NMIN, ns, work + NMIN, n, Bh, n - NMIN);
            CHECK_KERNEL();
            auto status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - NMIN, ns, n - NMIN,
                                       &sone, Ah, CUDA_R_16F, n - NMIN, Bh, CUDA_R_16F, n - NMIN,
                                       &szero, Ch, CUDA_R_16F, n - NMIN, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                    );
            ms = stopTimer();
            flops = 2.0f*(n-NMIN)*ns*(n-NMIN);
            total_ms = ms + total_ms;
            total_zy+=ms;
            total_flops = flops + total_flops;
            printf("matrix 1 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN, ns, n-NMIN,ms, flops/ms/1e9);
            //assert(status == CUBLAS_STATUS_SUCCESS);
            CHECK_KERNEL();
            Ah = hwork;
            startTimer();
            s2h<<<gridDimB,blockDimA>>>(n - ns, ns, A + ns, lda, Ah, n - ns);
            CHECK_KERNEL();
            status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n - NMIN, n - ns, ns,
                                       &snegone, Ch, CUDA_R_16F, n - NMIN, Ah, CUDA_R_16F, n - ns,
                                       &sone, A + NMIN + ns * lda, CUDA_R_32F, lda, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                    );

            ms = stopTimer();
            flops = 2.0f*(n-NMIN)*(n-ns)*ns;
            total_ms = ms + total_ms;
            total_zy+=ms;
            total_flops = flops + total_flops;
            printf("matrix 2 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN, n-ns, ns, ms, flops/ms/1e9);
            //assert(status == CUBLAS_STATUS_SUCCESS);
            CHECK_KERNEL();

            startTimer();
            dim3 gridDimC((n-NMIN+31)/32,(n-ns+31)/32);
            s2h<<<gridDimA,blockDimA>>>(n - NMIN, n - ns, A + NMIN + ns * lda, lda, Ch ,n - NMIN);
            status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, ns, n - ns, n - NMIN,
                                       &sone, Bh, CUDA_R_16F, n - NMIN, Ch, CUDA_R_16F, n - NMIN,
                                       &szero, Ah, CUDA_R_16F, ns, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                    );

            ms = stopTimer();
            flops = 2.0f*ns*(n-ns)*(n-NMIN);
            total_ms = ms + total_ms;
            total_flops = flops + total_flops;
            printf("matrix 3 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", ns, n-ns, n-NMIN, ms, flops/ms/1e9);
            startTimer();
            s2h<<<gridDimB,blockDimA>>>(n - ns, ns, A + ns, lda, Bh, n - ns);
            status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - ns, n - ns, ns,
                                       &snegone, Bh, CUDA_R_16F, n - ns, Ah, CUDA_R_16F, ns,
                                       &sone, A + ns + ns * NMIN, CUDA_R_32F, lda, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                    );
            ms = stopTimer();
            flops = 2.0f*ns*(n-ns)*(n-ns);
            total_ms = ms + total_ms;
            total_zy+=ms;
            total_flops = flops + total_flops;
            printf("matrix 4 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-ns, n-ns, ns,ms, flops/ms/1e9);
            printf("------Total time is %f ms, for size %d--------------------------------\n", total_ms, n);
            later_sy2sb_rec(ctxt, n - ns, ns, A+ns+ns*lda, oriA+ns+ns*lda, lda, work, lwork, hwork, lhwork);
        }
        else
        {
            if(i + NMIN >= ns)
                break;
            /*if this isn't the last iteration, update the next panel*/
            startTimer();
            __half *Ah = hwork;
            __half *Bh = hwork + (n - NMIN) * (n - NMIN);
            __half *Ch = hwork + (n - NMIN) * (n - NMIN) + (n - NMIN) * (i + NMIN);
            dim3 blockDimA(32,32);
            dim3 gridDimB((n-NMIN+31)/32,(i+NMIN+31)/32);
            s2h<<<gridDimB,blockDimA>>>(n - NMIN, i + NMIN, work + NMIN, n, Bh, n - NMIN);
            dim3 gridDimA((NMIN+31)/32,(i+NMIN+31)/32);
            s2h<<<gridDimA,blockDimA>>>(NMIN, i + NMIN, A + i + NMIN , lda, Ah ,NMIN);
            CHECK_KERNEL();
            auto status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n - NMIN, NMIN, i + NMIN,
                                       &sone, Bh, CUDA_R_16F, n - NMIN, Ah, CUDA_R_16F, NMIN,
                                       &szero, Ch, CUDA_R_16F, n - NMIN, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                    );
            
            ms = stopTimer();
            flops = 2.0f*NMIN*(i+NMIN)*(n-NMIN);
            total_ms = ms + total_ms;
            total_flops = flops + total_flops;
            printf("panel 1 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN, NMIN, i+NMIN, ms, flops/ms/1e9);
            //assert(status == CUBLAS_STATUS_SUCCESS);
            CHECK_KERNEL();
            startTimer();
            Ah = hwork;
            dim3 gridDimC((n-NMIN+31)/32,(n-NMIN+31)/32);
            s2h<<<gridDimC,blockDimA>>>(n - NMIN, n - NMIN, oriA + NMIN, lda, Ah ,n - NMIN);
            CHECK_KERNEL();
            status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - NMIN, NMIN, n - NMIN,
                                       &snegone, Ah, CUDA_R_16F, n - NMIN, Ch, CUDA_R_16F, n - NMIN,
                                       &sone, A + NMIN + (i + NMIN) * lda, CUDA_R_32F, lda, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                    );
            //assert(status == CUBLAS_STATUS_SUCCESS);
            ms = stopTimer();
            flops = 2.0f*(n-NMIN)*NMIN*(n-NMIN);
            total_ms = ms + total_ms;
            total_flops = flops + total_flops;
            printf("panel 2 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN, NMIN, n-NMIN, ms, flops/ms/1e9);
            CHECK_KERNEL();
            startTimer();
            dim3 gridDimD((n-NMIN+31)/32,(NMIN+31)/32);
            s2h<<<gridDimD,blockDimA>>>(n - NMIN, NMIN, A + NMIN + (i + NMIN) * lda, lda, Ah ,n - NMIN);
            status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, i + NMIN, NMIN, n - NMIN,
                                       &sone, Bh, CUDA_R_16F, n - NMIN, Ah, CUDA_R_16F, n - NMIN,
                                       &szero, Ch, CUDA_R_16F, i + NMIN, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                    );
            ms = stopTimer();
            flops = 2.0f*(i+NMIN)*NMIN*(n-NMIN);
            total_ms = ms + total_ms;
            total_flops = flops + total_flops;
            printf("panel 3 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", i+NMIN, NMIN, n-NMIN, ms, flops/ms/1e9);
            startTimer();
            dim3 gridDimE((n-NMIN-i+31)/32,(i+NMIN+31)/32);
            s2h<<<gridDimB,blockDimA>>>(n-NMIN-i, i + NMIN, A + i+ NMIN, lda, Ah, n - NMIN - i);
            status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - NMIN - i, NMIN, i + NMIN,
                                       &snegone, Ah, CUDA_R_16F, n - NMIN - i, Ch, CUDA_R_16F, i + NMIN,
                                       &sone, A + i + NMIN + (i + NMIN) * lda , CUDA_R_32F, lda, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                    );
            ms = stopTimer();
            flops = 2.0f*(n-NMIN-i)*NMIN*(i+NMIN);
            total_ms = ms + total_ms;
            total_flops = flops + total_flops;
            printf("panel 4 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN-i, NMIN, i+NMIN, ms, flops/ms/1e9);
        }
    }
    printf("------Total time is %f ms, zy is %fms, total flops is %.3e, rate is %f TFLOPS\n",total_ms, total_zy, total_flops, total_flops/total_ms/1e9);
    return; 
}