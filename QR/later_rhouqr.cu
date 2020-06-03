#include "LATER.h"
#include "LATER_QR.h"

#include <cuda_fp16.h>

#define NMIN 32

/*
This routine performs recursive Householder QR factorization

The input A stores the original matrix A to be factorized
The output A stores the Householder vectors Y
The output W stores the W matrix of WY representation
The orthogonal matrix Q 
THe output R stor
*/

void qr(cudaCtxt ctxt, int m, int n, float *A, int lda, float *W, int ldw, float *R, int ldr, float *work, int lwork,
    __half *hwork, int lhwork, float* U);


void reconstructY(cudaCtxt ctxt, int m,int n, float* dA, float *dU, int lda);

void checkError(int m,int n, float *A, int lda, float *W, int ldw, float *Y, int ldy , float *R, int ldr);

void later_rhouqr(int m, int n, float* A, int lda, float* W, int ldw, float* R, int ldr, float* work, int lwork, __half* hwork, int lhwork, float* U)
{
    printf("Function rhouqr\n");
    
    cudaCtxt ctxt;
    cublasCreate( & ctxt.cublas_handle );
    cusolverDnCreate( & ctxt.cusolver_handle );


    qr(ctxt, m, n, A, lda, W, ldw, R, ldr, work, lwork, hwork, lhwork, U);
    
    cublasDestroy(ctxt.cublas_handle);
    cusolverDnDestroy(ctxt.cusolver_handle);
    return;
}

__global__
void setZero(int m, int n, float *I, int ldi)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) 
    {
        I[i+j*ldi] = 0.0;
    }
}

void qr(cudaCtxt ctxt, int m, int n, float *A, int lda, float *W, int ldw, float *R, int ldr, float *work, int lwork, __half *hwork, int lhwork, float* U)
{
    if(n<=NMIN)
    {
        hou_caqr_panel<256,32,512>(ctxt, m, n, A, lda, R, ldr, work);
        dim3 gridDim((m+31)/32,(n+31)/32);
        dim3 blockDim(32,32);
        setEye<<<gridDim,blockDim>>>(m,n,W,ldw);
        sSubstract(ctxt.cublas_handle,m,n,A,lda,W,ldw);
        deviceCopy<<<gridDim,blockDim>>>( m, n, A, lda, W, ldw );

        reconstructY(ctxt,m,n,A,U,lda);

        float sone = 1.0;

        cublasStrsm(ctxt.cublas_handle,
            CUBLAS_SIDE_RIGHT,  CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T,  CUBLAS_DIAG_UNIT,
            m, n,
            &sone,
            A, lda,
            W, ldw
        );
        return;
    }
    
    qr(ctxt,m,n/2,A,lda,W,ldw,R,ldr,work,lwork,hwork,lhwork,U);
    float sone = 1.0, szero = 0.0,snegone = -1.0;

    if(n/2<=128 || m<=128)
    {
        cublasSgemm(ctxt.cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n/2,n/2,m,
            &sone,
            W, ldw,
            A+lda/2*n,lda,
            &szero,
            work,n/2
        );
        cublasSgemm(ctxt.cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m,n/2,n/2,
            &snegone,
            A, lda,
            work,n/2,
            &sone,
            A+lda/2*n,lda
        );
    }
    else
    {
        __half *Ah = hwork;
        __half *Bh = hwork+m/2*n;
        startTimer();
        dim3 gridDimA((m+31)/32,(n/2+31)/32);
        dim3 blockDimA(32,32);
        s2h<<<gridDimA,blockDimA>>>(m,n/2,W,ldw,Ah,m);
        s2h<<<gridDimA,blockDimA>>>(m,n/2,A+lda/2*n,lda,Bh,m);
        cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n/2, n/2, m,
            &sone, Ah, CUDA_R_16F, m, Bh, CUDA_R_16F, m,
            &szero, work, CUDA_R_32F, n/2, CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );

        s2h<<<gridDimA,blockDimA>>>(m,n/2,A,lda,Ah,m);

        dim3 gridDimB((n/2+31)/32,(n/2+31)/32);
        dim3 blockDimB(32,32);
        s2h<<<gridDimB,blockDimB>>>(n/2,n/2,work,n/2,Bh,n/2);
        cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n/2, n/2,
            &snegone, Ah, CUDA_R_16F, m, Bh, CUDA_R_16F, n/2,
            &sone, A+lda/2*n, CUDA_R_32F, lda, CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }

    qr(ctxt,m-n/2,n/2,A+lda/2*n+n/2, lda, W+ldw/2*n+n/2, ldw, R+n/2*ldr+n/2, ldr,work, lwork,hwork,lhwork,U);

    dim3 gridDim1((n/2+31)/32,(n/2+31)/32);
    dim3 blockDim1(32,32);

    deviceCopy<<<gridDim1,blockDim1>>>(n/2, n/2, A+lda/2*n, lda, R+ldr/2*n, ldr);

    setZero<<<gridDim1,blockDim1>>>(n/2,n/2,A+lda/2*n,lda);

    if(n/2<=128 || m<=128)
    {
        cublasSgemm(ctxt.cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n/2,n/2,m,
            &sone,
            A, lda,
            W+ldw/2*n,ldw,
            &szero,
            work,n/2
        );
        cublasSgemm(ctxt.cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m,n/2,n/2,
            &snegone,
            W, ldw,
            work,n/2,
            &sone,
            W+ldw/2*n,ldw
        );
    }
    else
    {
        __half *Ah = hwork;
        __half *Bh = hwork+m/2*n;
        startTimer();
        dim3 gridDimA((m+31)/32,(n/2+31)/32);
        dim3 blockDimA(32,32);
        s2h<<<gridDimA,blockDimA>>>(m,n/2,A,lda,Ah,m);
        s2h<<<gridDimA,blockDimA>>>(m,n/2,W+ldw/2*n,ldw,Bh,m);
        cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n/2, n/2, m,
            &sone, Ah, CUDA_R_16F, m, Bh, CUDA_R_16F, m,
            &szero, work, CUDA_R_32F, n/2, CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );

        s2h<<<gridDimA,blockDimA>>>(m,n/2,W,ldw,Ah,m);

        dim3 gridDimB((n/2+31)/32,(n/2+31)/32);
        dim3 blockDimB(32,32);
        s2h<<<gridDimB,blockDimB>>>(n/2,n/2,work,n/2,Bh,n/2);
        cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n/2, n/2,
            &snegone, Ah, CUDA_R_16F, m, Bh, CUDA_R_16F, n/2,
            &sone, W+ldw/2*n, CUDA_R_32F, ldw, CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }
    return;
}

// get U from LU factorization
__global__
void getU(int m, int n, float *a, int lda, float *u, int ldr)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i<m && j<n) 
    {
        if (i>j)
            u[i+j*ldr]  = 0;
        else 
            u[i+j*ldr] = a[i+j*lda];
	}
}

// get L from LU factorization
__global__
void getL(int m, int n, float *a, int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i<m && j<n) 
    {
        if (i<j)
            a[i+j*lda] = 0;
        else if (i==j)
            a[i+j*lda] = 1;
	}
}

void reconstructY(cudaCtxt ctxt, int m,int n, float* dA, float *dU, int lda)
{
    int *d_info = NULL; /* error info */
    int  lwork = 0;     /* size of workspace */
    float *d_work = NULL; /* device workspace for getrf */
    
    cusolverDnSgetrf_bufferSize(
        ctxt.cusolver_handle,
        n,
        n,
        dA,
        lda,
        &lwork
    );

    cudaMalloc ((void**)&d_info, sizeof(int));
    cudaMalloc((void**)&d_work, sizeof(float)*lwork);

    cusolverDnSgetrf(
        ctxt.cusolver_handle,
        n,
        n,
        dA,
        lda,
        d_work,
        NULL,
        d_info
    );

    dim3 gridDim((n+31)/32,(n+31)/32);
    dim3 blockDim(32,32);
    getU<<<gridDim,blockDim>>>(n,n,dA,lda,dU,n);
    getL<<<gridDim, blockDim>>>(n,n,dA,lda);

    float sone = 1.0;
    cublasStrsm(ctxt.cublas_handle,
        CUBLAS_SIDE_RIGHT,  CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N,  CUBLAS_DIAG_NON_UNIT,
        m-n, n,
        &sone,
        dU, n,
        dA+n, lda
    );
}

__global__
void clear_tri(char uplo, int m, int n, float *a, int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		if (uplo == 'l') {
			if (i>j) {
				a[i+j*lda] = 0;
			}
        } 
        else
        {
            if (i<j)
                a[i+j*lda] = 0;
		}
	}
}

void checkError(int m,int n, float *A, int lda, float *W, int ldw, float *Y, int ldy , float *R, int ldr)
{
    float *I;
    cudaMalloc(&I,sizeof(float)*n*n);
      
	dim3 grid96( (n+1)/32, (n+1)/32 );
	dim3 block96( 32, 32 );
    setEye<<<grid96,block96>>>( n, n, I, n);
    float snegone = -1.0;
    float sone  = 1.0;

    float *WI;
    cudaMalloc(&WI, sizeof(float)*m*n);
    dim3 grid1( (m+1)/32, (n+1)/32 );
	dim3 block1( 32, 32 );
    setEye<<<grid1,block1>>>( m, n, WI, m);

    clear_tri<<<grid96,block96>>>('l',n,n,R,ldr);   
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,m,n,n,
        &snegone,W,CUDA_R_32F, ldw, Y, CUDA_R_32F, ldy,
        &sone, WI, CUDA_R_32F, m, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    );

    
    
    float normWI= snorm(m,n,WI);
    printf("normWI = %f\n", normWI);
    

    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
        &snegone, WI, CUDA_R_32F, m, WI, CUDA_R_32F, m,
        &sone, I, CUDA_R_32F, n, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT);
    
    float normRes = snorm(n,n,I);
   
    printf("||I-Q'*Q||/N = %.6e\n",normRes/n);

    //printMatrixDeviceBlock("AA.csv",m,n,A,lda);
    float normA = snorm(m,n,A);
    printf("normA = %f\n", normA);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
        &snegone, WI, CUDA_R_32F, m, R, CUDA_R_32F, ldr,
        &sone, A, CUDA_R_32F, lda, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
        );
    normRes = snorm(m,n,A);
    printf("||A-QR||/||A|| = %.6e\n",normRes/normA);
    cudaFree(I);
    cudaFree(WI);
    cublasDestroy(handle);
}
