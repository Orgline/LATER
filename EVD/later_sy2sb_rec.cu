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
	if (i<m && j<n) 
    {
        int block = j/NMIN;
        int tmp = (block+1)*NMIN;
        if(i >= (block+2)*NMIN)
            dA[i+j*lda] = 0;
        else if(i >= tmp)
        {
            dA[i+j*lda] = dR[(i-tmp)+j*ldr];
        }        
	}
}

float flops = 0.0;
float total_flops = 0.0;
float ms =0.0;
float total_ms = 0.0;
float total_zy = 0.0;
float total_qr = 0.0;
bool sgemm_flag = true;

__global__
void s2hAndClearTri(int m, int n, float *as, int ldas, __half *ah, int ldah)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
        if(j>i)
            ah[i + j*ldah] = __float2half(0.0f);
        else
		    ah[i + j*ldah] = __float2half(as[i + j*ldas]);
	}
}

__global__
void s2sAndClearTri(int m, int n, float *as, int ldas, float *ah, int ldah)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
        if(j>i)
            ah[i + j*ldah] = 0.0f;
        else
		    ah[i + j*ldah] = as[i + j*ldas];
	}
}


void later_sy2sb_rec(cudaCtxt ctxt, int n, int ns, float *A, float *oriA, int lda, float *work, int lwork, __half *hwork, int lhwork)
{
    printf("n = %d\n", n);
    
    float sone = 1.0f;
    float snegone = -1.0f;
    float szero = 0.0f;
    int end_ind;
    if(ns == n)
        end_ind = n - NMIN;
    else
        end_ind = ns;
    for(int i = 0; i < end_ind; i+=NMIN)
    {
        int lwork = n*NMIN;
        int lhwork = n*NMIN;
        startTimer();
        /*
        work to work[ns*lda] is used for storing W matirx;
        work[ns*lda] to work[ns*lda+lda*NMIN] is used for storing R matrix; 
        */
        //printMatrixDeviceBlock("A.csv", n-i-NMIN, NMIN, A + i+NMIN+i*lda, lda);
        later_rhouqr(ctxt, n - i - NMIN, NMIN, &A[i+NMIN+i*lda], lda, work+i+NMIN+i*n, n, work+ns*lda+i*NMIN, NMIN, work+ns*lda+ns*NMIN, lwork, hwork, lhwork, work+ns*lda+ns*NMIN+lwork+lwork);
        //later_ormqr(n-i-NMIN, NMIN, work+i+NMIN+i*n, n, A + i+NMIN+i*lda, lda, work+ns*n+ns*NMIN);
        ms = stopTimer();
        //printf("QR size %d*%d, takes %f ms\n",n-i-NMIN, NMIN, ms);
        total_qr += ms;
        //total_ms += ms;
        //checkOtho_(n-i-NMIN, NMIN, work+i+NMIN+i*n, n);
        //checkResult(n-i-NMIN, NMIN, oriA+ i+NMIN+i*lda, lda, work+i+NMIN+i*n, n,  work+ns*n+i*NMIN, NMIN);
        
        //printMatrixDeviceBlock("Y1.csv", n-i-NMIN, NMIN, A + i+NMIN+i*lda, lda);
        
        //printMatrixDeviceBlock("W1.csv", n-i-NMIN, NMIN, work+i+NMIN+i*n, n);
        //printMatrixDeviceBlock("R.csv", NMIN, NMIN, work+ns*n+i*NMIN, NMIN);

        if(i > 0)
        {
            if(!sgemm_flag)
            {
                /*form the new W matrix*/
                startTimer();
                __half *Ah = hwork;
                __half *Bh = hwork + i * (n - NMIN);
                __half *Ch = hwork + i * (n - NMIN) + NMIN * (n - NMIN);
                
                dim3 gridDimA((n-NMIN+31)/32,(i+31)/32);
                dim3 blockDimA(32,32);
                s2hAndClearTri<<<gridDimA,blockDimA>>>(n - NMIN, i, A + NMIN, lda , Ah , n - NMIN);
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
                //printf("Form W 1 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", i, NMIN, n-NMIN, ms, 2.0f*i*NMIN*(n-NMIN)/ms/1e9);
                //assert(status == CUBLAS_STATUS_SUCCESS);
                CHECK_KERNEL();

                startTimer();
                s2h<<<gridDimA,blockDimA>>>(n - NMIN, i, work + NMIN, n, Ah, n - NMIN);
                CHECK_KERNEL();
                status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - NMIN, NMIN, i,
                                        &snegone, Ah, CUDA_R_16F, n - NMIN, Ch, CUDA_R_16F, i,
                                        &sone, work+NMIN+i*n, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                                        CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                        );
                ms = stopTimer();
                // if(ns == n){
                // printMatrixDeviceBlock("WW.csv", n-NMIN, ns, work+NMIN, n);
                // printMatrixDeviceBlock("YY.csv", n-NMIN, ns, A+NMIN, lda);
                // return;}

                flops = 2.0f*i*NMIN*(n-NMIN);
                total_ms = ms + total_ms;
                total_flops = flops + total_flops;
                //printf("Form w 2 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n - NMIN, NMIN, i, ms, flops/ms/1e9);
                //return;
                //assert(status == CUBLAS_STATUS_SUCCESS);
                CHECK_KERNEL();
            }
            else
            {
                /*form the new W matrix*/
                startTimer();
                float *Ah = work+ns*lda+ns*NMIN;
                float *Ch = work + i * (n - NMIN);
                
                dim3 gridDimA((n-NMIN+31)/32,(i+31)/32);
                dim3 blockDimA(32,32);
                s2sAndClearTri<<<gridDimA,blockDimA>>>(n - NMIN, i, A + NMIN, lda , Ah , n - NMIN);
                
                CHECK_KERNEL();
                
                auto status = cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, i, NMIN, n - NMIN,
                                        &sone, Ah, n - NMIN, work+ NMIN + i * n, n,
                                        &szero, Ch, i);

                ms = stopTimer();
                flops = 2.0f*i*NMIN*(n-NMIN);
                total_ms = ms + total_ms;
                total_flops = flops + total_flops;
                //printf("Form W 1 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", i, NMIN, n-NMIN, ms, 2.0f*i*NMIN*(n-NMIN)/ms/1e9);
                //assert(status == CUBLAS_STATUS_SUCCESS);
                CHECK_KERNEL();

                startTimer();
                
                CHECK_KERNEL();
                status = cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - NMIN, NMIN, i,
                                        &snegone, work + NMIN, n, Ch, i,
                                        &sone, work+NMIN+i*n, n);
                ms = stopTimer();
                // if(ns == n){
                // printMatrixDeviceBlock("WW.csv", n-NMIN, ns, work+NMIN, n);
                // printMatrixDeviceBlock("YY.csv", n-NMIN, ns, A+NMIN, lda);
                // return;}

                flops = 2.0f*i*NMIN*(n-NMIN);
                total_ms = ms + total_ms;
                total_flops = flops + total_flops;
                //printf("Form w 2 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n - NMIN, NMIN, i, ms, flops/ms/1e9);
                //return;
                //assert(status == CUBLAS_STATUS_SUCCESS);
                CHECK_KERNEL();
            }
        }
        
        if(i + NMIN >= ns && n > ns)
        {
            if(!sgemm_flag)
            {
                /*if this is the last iteration, update the whole trailing matrix*/
                startTimer();
                __half *Ah = hwork;
                __half *Bh = hwork + (n - NMIN) * (n - NMIN);
                __half *Ch = hwork + (n - NMIN) * (n - NMIN) + (n - NMIN) * ns;
                dim3 gridDimA((n-NMIN+31)/32,(n-NMIN+31)/32);
                dim3 blockDimA(32,32);
                s2h<<<gridDimA,blockDimA>>>(n - NMIN, n - NMIN, oriA + NMIN + NMIN * lda, lda, Ah ,n-NMIN);
                dim3 gridDimB((n-NMIN+31)/32,(ns+31)/32);
                s2h<<<gridDimB,blockDimA>>>(n - NMIN, ns, work + NMIN, n, Bh, n - NMIN);
                CHECK_KERNEL();
                //printMatrixDeviceBlock("oriA.csv", n-NMIN, n-NMIN, oriA+NMIN+NMIN * lda, lda);
                //printMatrixDeviceBlock("Wmid.csv", n-NMIN, ns, work+NMIN, n);
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
                //printf("matrix 1 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN, ns, n-NMIN,ms, flops/ms/1e9);
                //assert(status == CUBLAS_STATUS_SUCCESS);
                CHECK_KERNEL();
                Ah = hwork;
                startTimer();
                dim3 gridDimC((n-ns+31)/32,(ns+31)/32);
                s2h<<<gridDimC,blockDimA>>>(n - ns, ns, A + ns, lda, Ah, n - ns);
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
                //printf("matrix 2 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN, n-ns, ns, ms, flops/ms/1e9);
                //printMatrixDeviceBlock("Amid.csv", n - NMIN, n - ns, A + NMIN + ns * lda, lda);
                //return;
                //assert(status == CUBLAS_STATUS_SUCCESS);
                CHECK_KERNEL();

                startTimer();
                dim3 gridDimD((n-NMIN+31)/32,(n-ns+31)/32);
                s2h<<<gridDimD,blockDimA>>>(n - NMIN, n - ns, A + NMIN + ns * lda, lda, Ch ,n - NMIN);
                status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, ns, n - ns, n - NMIN,
                                        &sone, Bh, CUDA_R_16F, n - NMIN, Ch, CUDA_R_16F, n - NMIN,
                                        &szero, Ah, CUDA_R_16F, ns, CUBLAS_COMPUTE_32F,
                                        CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                        );

                ms = stopTimer();
                flops = 2.0f*ns*(n-ns)*(n-NMIN);
                total_ms = ms + total_ms;
                total_flops = flops + total_flops;
                //printf("matrix 3 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", ns, n-ns, n-NMIN, ms, flops/ms/1e9);
                startTimer();
                s2h<<<gridDimC,blockDimA>>>(n - ns, ns, A + ns, lda, Bh, n - ns);
                status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - ns, n - ns, ns,
                                        &snegone, Bh, CUDA_R_16F, n - ns, Ah, CUDA_R_16F, ns,
                                        &sone, A + ns + ns * lda, CUDA_R_32F, lda, CUBLAS_COMPUTE_32F,
                                        CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                        );
                ms = stopTimer();
                flops = 2.0f*ns*(n-ns)*(n-ns);
                total_ms = ms + total_ms;
                total_zy+=ms;
                total_flops = flops + total_flops;
                //printf("matrix 4 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-ns, n-ns, ns,ms, flops/ms/1e9);
                //printf("------Total time is %f ms, for size %d--------------------------------\n", total_ms, n);
                //printMatrixDeviceBlock("newA.csv", lda, lda, A, lda);
                //printMatrixDeviceBlock("newR.csv", NMIN, lda, work+ns*lda, NMIN);
                dim3 gridDimE((n+31)/32,(ns+31)/32);
                copyRtoPanel<<<gridDimE, blockDimA>>>(n, ns, A, lda, work+ns*lda, NMIN);
                dim3 gridDimF((n -ns+31)/32,(n-ns+31)/32);
                deviceCopy<<<gridDimF, blockDimA>>>(n-ns, n-ns, A+ns+ns*lda, lda, oriA+ns+ns*lda, lda);
                cudaMemset(work, 0, sizeof(float)*2*lda*lda);
            }
            else
            {
                startTimer();
                float *Ah = work+ns*lda+ns*NMIN;
                
                float *Ch = Ah + (n - NMIN) * (n - NMIN);
                dim3 gridDimA((n-NMIN+31)/32,(n-NMIN+31)/32);
                dim3 blockDimA(32,32);
                //s2h<<<gridDimA,blockDimA>>>(n - NMIN, n - NMIN, oriA + NMIN + NMIN * lda, lda, Ah ,n-NMIN);
                dim3 gridDimB((n-NMIN+31)/32,(ns+31)/32);
                //s2h<<<gridDimB,blockDimA>>>(n - NMIN, ns, work + NMIN, n, Bh, n - NMIN);
                CHECK_KERNEL();
                //printMatrixDeviceBlock("oriA.csv", n-NMIN, n-NMIN, oriA+NMIN+NMIN * lda, lda);
                //printMatrixDeviceBlock("Wmid.csv", n-NMIN, ns, work+NMIN, n);
                auto status = cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - NMIN, ns, n - NMIN,
                                        &sone, oriA + NMIN + NMIN * lda, lda, work + NMIN, n,
                                        &szero, Ch, n - NMIN);
                ms = stopTimer();
                flops = 2.0f*(n-NMIN)*ns*(n-NMIN);
                total_ms = ms + total_ms;
                total_zy+=ms;
                total_flops = flops + total_flops;
                //printf("matrix 1 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN, ns, n-NMIN,ms, flops/ms/1e9);
                //assert(status == CUBLAS_STATUS_SUCCESS);
                CHECK_KERNEL();
                //Ah = hwork;
                startTimer();
                dim3 gridDimC((n-ns+31)/32,(ns+31)/32);
                //s2h<<<gridDimC,blockDimA>>>(n - ns, ns, A + ns, lda, Ah, n - ns);
                CHECK_KERNEL();
                status = cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n - NMIN, n - ns, ns,
                                        &snegone, Ch, n - NMIN,  A + ns, lda,
                                        &sone, A + NMIN + ns * lda,  lda);

                ms = stopTimer();
                flops = 2.0f*(n-NMIN)*(n-ns)*ns;
                total_ms = ms + total_ms;
                total_zy+=ms;
                total_flops = flops + total_flops;
                //printf("matrix 2 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN, n-ns, ns, ms, flops/ms/1e9);
                //printMatrixDeviceBlock("Amid.csv", n - NMIN, n - ns, A + NMIN + ns * lda, lda);
                //return;
                //assert(status == CUBLAS_STATUS_SUCCESS);
                CHECK_KERNEL();

                startTimer();
                dim3 gridDimD((n-NMIN+31)/32,(n-ns+31)/32);
                //s2h<<<gridDimD,blockDimA>>>(n - NMIN, n - ns, A + NMIN + ns * lda, lda, Ch ,n - NMIN);
                status = cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, ns, n - ns, n - NMIN,
                                        &sone, work + NMIN, n, A + NMIN + ns * lda, lda,
                                        &szero, Ah, ns);

                ms = stopTimer();
                flops = 2.0f*ns*(n-ns)*(n-NMIN);
                total_ms = ms + total_ms;
                total_flops = flops + total_flops;
                //printf("matrix 3 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", ns, n-ns, n-NMIN, ms, flops/ms/1e9);
                startTimer();
                //s2h<<<gridDimC,blockDimA>>>(n - ns, ns, A + ns, lda, Bh, n - ns);
                status = cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - ns, n - ns, ns,
                                        &snegone, A + ns, lda, Ah, ns,
                                        &sone, A + ns + ns * lda, lda);
                ms = stopTimer();
                flops = 2.0f*ns*(n-ns)*(n-ns);
                total_ms = ms + total_ms;
                total_zy+=ms;
                total_flops = flops + total_flops;
                //printf("matrix 4 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-ns, n-ns, ns,ms, flops/ms/1e9);
                //printf("------Total time is %f ms, for size %d--------------------------------\n", total_ms, n);
                //printMatrixDeviceBlock("newA.csv", lda, lda, A, lda);
                //printMatrixDeviceBlock("newR.csv", NMIN, lda, work+ns*lda, NMIN);
                dim3 gridDimE((n+31)/32,(ns+31)/32);
                copyRtoPanel<<<gridDimE, blockDimA>>>(n, ns, A, lda, work+ns*lda, NMIN);
                dim3 gridDimF((n -ns+31)/32,(n-ns+31)/32);
                deviceCopy<<<gridDimF, blockDimA>>>(n-ns, n-ns, A+ns+ns*lda, lda, oriA+ns+ns*lda, lda);
                cudaMemset(work, 0, sizeof(float)*2*lda*lda);
            }
            //printMatrixDeviceBlock("newA_.csv", lda, lda, A, lda);
            //printMatrixDeviceBlock("newR_.csv", NMIN, lda, work+ns*lda, NMIN);
            //return;
            later_sy2sb_rec(ctxt, n - ns, ns, A+ns+ns*lda, oriA+ns+ns*lda, lda, work, lwork, hwork, lhwork);
        }
        else
        {
            /*if this is the last iteration, the */
            // if(i + NMIN >= ns)
            //     break;
            /*if this isn't the last iteration, update the next panel*/
            if(!sgemm_flag)
            {
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
                //printMatrixDeviceBlock("Y.csv", NMIN, i+NMIN, A + i + NMIN, lda);
                //printMatrixDeviceBlock("W.csv", n-NMIN, i+NMIN, work + NMIN, n);
                auto status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n - NMIN, NMIN, i + NMIN,
                                        &sone, Bh, CUDA_R_16F, n - NMIN, Ah, CUDA_R_16F, NMIN,
                                        &szero, Ch, CUDA_R_16F, n - NMIN, CUBLAS_COMPUTE_32F,
                                        CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                        );
                
                ms = stopTimer();
                flops = 2.0f*NMIN*(i+NMIN)*(n-NMIN);
                total_ms = ms + total_ms;
                total_flops = flops + total_flops;
                //printf("panel 1 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN, NMIN, i+NMIN, ms, flops/ms/1e9);
                //assert(status == CUBLAS_STATUS_SUCCESS);
                CHECK_KERNEL();
                startTimer();
                Ah = hwork;
                dim3 gridDimC((n-NMIN+31)/32,(n-NMIN+31)/32);
                s2h<<<gridDimC,blockDimA>>>(n - NMIN, n - NMIN, oriA + NMIN+ NMIN * lda, lda, Ah ,n - NMIN);
                CHECK_KERNEL();
                //printMatrixDeviceBlock("old_panel.csv", n-NMIN, NMIN, A + NMIN + (i + NMIN) * lda, lda);
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
                //printf("panel 2 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN, NMIN, n-NMIN, ms, flops/ms/1e9);
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
                //printf("panel 3 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", i+NMIN, NMIN, n-NMIN, ms, flops/ms/1e9);
                //printMatrixDeviceBlock("Y.csv", n - NMIN - i, i+NMIN, A + i+ NMIN, lda);
                startTimer();
                dim3 gridDimE((n-NMIN-i+31)/32,(i+NMIN+31)/32);
                s2h<<<gridDimB,blockDimA>>>(n-NMIN-i, i + NMIN, A + i + NMIN, lda, Ah, n - NMIN - i);
                status = cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - NMIN - i, NMIN, i + NMIN,
                                        &snegone, Ah, CUDA_R_16F, n - NMIN - i, Ch, CUDA_R_16F, i + NMIN,
                                        &sone, A + i + NMIN + (i + NMIN) * lda , CUDA_R_32F, lda, CUBLAS_COMPUTE_32F,
                                        CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                        );
                ms = stopTimer();
                flops = 2.0f*(n-NMIN-i)*NMIN*(i+NMIN);
                total_ms = ms + total_ms;
                total_flops = flops + total_flops;
                if(n == ns && i+NMIN >= end_ind)
                {
                printf("last panel\n");
                dim3 gridDimF((n+31)/32,(n-NMIN+31)/32);
                copyRtoPanel<<<gridDimF, blockDimA>>>(n, n - NMIN, A, lda, work+ns*lda, NMIN);
                }
            }
            else
            {
                startTimer();
                float *Ch = work+ns*lda+ns*NMIN;
                dim3 blockDimA(32,32);
                dim3 gridDimB((n-NMIN+31)/32,(i+NMIN+31)/32);
                //s2h<<<gridDimB,blockDimA>>>(n - NMIN, i + NMIN, work + NMIN, n, Bh, n - NMIN);
                dim3 gridDimA((NMIN+31)/32,(i+NMIN+31)/32);
                //s2h<<<gridDimA,blockDimA>>>(NMIN, i + NMIN, A + i + NMIN , lda, Ah ,NMIN);
                //CHECK_KERNEL();
                //printMatrixDeviceBlock("Y.csv", NMIN, i+NMIN, A + i + NMIN, lda);
                //printMatrixDeviceBlock("W.csv", n-NMIN, i+NMIN, work + NMIN, n);
                auto status = cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n - NMIN, NMIN, i + NMIN,
                                        &sone, work + NMIN, n, A + i + NMIN, lda,
                                        &szero, Ch, n - NMIN);
                
                ms = stopTimer();
                flops = 2.0f*NMIN*(i+NMIN)*(n-NMIN);
                total_ms = ms + total_ms;
                total_flops = flops + total_flops;
                //printf("panel 1 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN, NMIN, i+NMIN, ms, flops/ms/1e9);
                //assert(status == CUBLAS_STATUS_SUCCESS);
                CHECK_KERNEL();
                startTimer();
                //Ah = hwork;
                dim3 gridDimC((n-NMIN+31)/32,(n-NMIN+31)/32);
                //s2h<<<gridDimC,blockDimA>>>(n - NMIN, n - NMIN, oriA + NMIN+ NMIN * lda, lda, Ah ,n - NMIN);
                //CHECK_KERNEL();
                //printMatrixDeviceBlock("old_panel.csv", n-NMIN, NMIN, A + NMIN + (i + NMIN) * lda, lda);
                status = cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - NMIN, NMIN, n - NMIN,
                                        &snegone, oriA + NMIN+ NMIN * lda, lda, Ch, n - NMIN,
                                        &sone, A + NMIN + (i + NMIN) * lda, lda);
                
                //assert(status == CUBLAS_STATUS_SUCCESS);
                ms = stopTimer();
                flops = 2.0f*(n-NMIN)*NMIN*(n-NMIN);
                total_ms = ms + total_ms;
                total_flops = flops + total_flops;
                //printf("panel 2 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN, NMIN, n-NMIN, ms, flops/ms/1e9);
                CHECK_KERNEL();
                startTimer();
                dim3 gridDimD((n-NMIN+31)/32,(NMIN+31)/32);
                //s2h<<<gridDimD,blockDimA>>>(n - NMIN, NMIN, A + NMIN + (i + NMIN) * lda, lda, Ah ,n - NMIN);
                status = cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, i + NMIN, NMIN, n - NMIN,
                                        &sone, work + NMIN, n, A + NMIN + (i + NMIN) * lda, n - NMIN,
                                        &szero, Ch, i + NMIN);
                ms = stopTimer();
                flops = 2.0f*(i+NMIN)*NMIN*(n-NMIN);
                total_ms = ms + total_ms;
                total_flops = flops + total_flops;
                //printf("panel 3 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", i+NMIN, NMIN, n-NMIN, ms, flops/ms/1e9);
                //printMatrixDeviceBlock("Y.csv", n - NMIN - i, i+NMIN, A + i+ NMIN, lda);
                startTimer();
                dim3 gridDimE((n-NMIN-i+31)/32,(i+NMIN+31)/32);
                //s2h<<<gridDimB,blockDimA>>>(n-NMIN-i, i + NMIN, A + i + NMIN, lda, Ah, n - NMIN - i);
                status = cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - NMIN - i, NMIN, i + NMIN,
                                        &snegone, A + i + NMIN, lda, Ch, i + NMIN,
                                        &sone, A + i + NMIN + (i + NMIN) * lda, lda);
                ms = stopTimer();
                flops = 2.0f*(n-NMIN-i)*NMIN*(i+NMIN);
                total_ms = ms + total_ms;
                total_flops = flops + total_flops;
                if(n == ns && i+NMIN >= end_ind)
                {
                printf("last panel\n");
                dim3 gridDimF((n+31)/32,(n-NMIN+31)/32);
                copyRtoPanel<<<gridDimF, blockDimA>>>(n, n - NMIN, A, lda, work+ns*lda, NMIN);
                }
            }
            //printMatrixDeviceBlock("new_panel.csv", n - NMIN - i, NMIN, A + i + NMIN + (i + NMIN) * lda, lda);
            //printf("panel 4 GEMM size is %d*%d*%d takes %fms, rate is %f TFLOPs\n", n-NMIN-i, NMIN, i+NMIN, ms, flops/ms/1e9);
            // if(n == ns && i == NMIN){
            //     printMatrixDeviceBlock("new_panel.csv", n-i-NMIN, NMIN, A + i+NMIN + (i + NMIN) * lda, lda);
            //     return;
            // }
            

            //return;
        }
    }
    printf("------Total time is %f ms, qr is %fms, total zy is %.3e, rate is %f TFLOPS\n",total_ms, total_qr, total_zy, total_flops/total_ms/1e9);
    return; 
}