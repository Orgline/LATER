#include "LATER.h"
#include "LATER_QR.h"
#include <assert.h>
#include <string>



#define NMIN 128

int algo;
int m,n;

//for rgsqrf
void checkResult(int m,int n,float* A,int lda, float *Q, int ldq, float *R, int ldr);
//for rhouqr
void checkResult(int m,int n, float *A, int lda, float *W, int ldw, float *Y, int ldy , float *R, int ldr);
void sgemm(int m,int n,int k,float *dA,int lda, float *dB,int ldb,float *dC, int ldc,float alpha,float beta);
void checkOtho(int,int,float*, int);

bool checkFlag = false;

int parseArguments(int argc,char *argv[])
{
    algo = atoi(argv[1]);
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    for (int i=4; i<argc; i++) {
        if(strcmp(argv[i], "-check") == 0) {
            checkFlag = true;
        }
    }
    return 0;
}

int main(int argc,char *argv[])
{
    if (argc < 4) {
        printf("Usage: test algo m n [options]\n");
        printf("Options:\n\t-check: enable checking the orthogonality and backward error\n");
        return 0;
    }
    if(parseArguments(argc,argv)!=0)
    {
        return 0;
    }
    print_env();
    float *A;
    cudaMalloc(&A,sizeof(float)*m*n);
    float *R;
    cudaMalloc(&R,sizeof(float)*n*n);

    generateUniformMatrix(A,m,n);

    //printMatrixDeviceBlock("A.csv",m,n,A,m);


    float *dA;

    cudaCtxt ctxt {};
    cublasCreate(&ctxt.cublas_handle );
    cusolverDnCreate(&ctxt.cusolver_handle );

    int lwork;

    


        
    if (algo == 1)
    {

        //int lwork = (n/2)*(n/2);
        int lwork = m/256*32*n;
        float *work;
        __half *hwork;
        int lhwork = m*n;
        cudaMalloc( &hwork, sizeof(__half) * lhwork );
        cudaMalloc( &work, sizeof(float)*lwork );
        printf("Perform RGSQRF\nmatrix size %d*%d\n",m,n);
        startTimer();

        later_rgsqrf(ctxt, m,n,A,m,R,n,work,lwork,hwork,lhwork);

        float ms = stopTimer();
        printf("RGSQRF takes %.0f ms, exec rate %.0f GFLOPS\n", ms, 
                2.0*n*n*( m -1.0/3.0*n )/(ms*1e6));

        if (checkFlag) {


            checkOtho(m, n, A, m);

            cudaMalloc(&dA,sizeof(float)*m*n);
            generateUniformMatrix(dA,m,n);

            checkResult(m, n, dA, m, A, m, R, n);
            cudaFree(dA);
        }
        cudaFree(work);
        cudaFree(hwork);
    }

    if (algo == 2)
    {
        printf("Perform RHOUQR\nmatrix size %d*%d\n",m,n);
        __half *hwork;
        int lhwork = m*n;
        cudaMalloc( &hwork, sizeof(__half) * lhwork );
        float *U;
        cudaMalloc(&U,sizeof(float)*32*32);
        float *W;
        cudaMalloc(&W,sizeof(float)*m*n);

        float *work;
        int lwork = m/256*32*n;
        cudaMalloc(&work, sizeof(float)*lwork);

        startTimer();
        later_rhouqr(m, n, A, m, W, m, R, n, work, lwork, hwork, lhwork, U);
        float ms = stopTimer();
        printf("RHOUQR takes %.0f ms, exec rate %.0f GFLOPS\n", ms, 
                2.0*n*n*( m -1.0/3.0*n )/(ms*1e6));

        if(checkFlag)
        {
            cudaMalloc(&dA,sizeof(float)*m*n);
            generateUniformMatrix(dA,m,n);
            later_ormqr(m, n, W, m, A, m, work);
            //printMatrixDeviceBlock("Q.csv",m,n,W,m);
            //checkResult( m, n, dA, m, W, m, A, m , R, n );
            checkOtho(m, n, W, m);
            dim3 grid96( (n+1)/32, (n+1)/32 );
            dim3 block96( 32, 32 );
            clearTri<<<grid96,block96>>>('l',n,n,R,n);  
            //printMatrixDeviceBlock("R.csv",n,n,R,n);
            checkResult(m, n, dA, m, W, m, R, n);
            cudaFree(dA);
        }

        cudaFree(U);
        cudaFree(W);
        cudaFree(work);
        cudaFree(hwork);
    }

    if(algo == 3)
    {
        printf("Perform BHOUQR\nmatrix size %d*%d\n",m,n);
        __half *hwork;
        int lhwork = m*n;
        cudaMalloc( &hwork, sizeof(__half) * lhwork );
        float *U;
        cudaMalloc(&U,sizeof(float)*32*32);
        float *W;
        cudaMalloc(&W,sizeof(float)*m*n);

        float *work;
        int lwork = m*n;
        cudaMalloc(&work, sizeof(float)*lwork);

        //startTimer();
        later_bhouqr(m, n, A, m, W, m, R, n, work, lwork, hwork, lhwork, U);
        //float ms = stopTimer();
        //printf("BHOUQR takes %.0f ms, exec rate %.0f GFLOPS\n", ms, 
                //2.0*n*n*( m -1.0/3.0*n )/(ms*1e6));
        
        if(checkFlag)
        {
            cudaMalloc(&dA,sizeof(float)*m*n);
            generateUniformMatrix(dA,m,n);
            later_ormqr2(m, n, W, m, A, m, work);
            checkOtho(m, n, W, m);
            checkResult(m, n, dA, m, W, m, R, n);
            cudaFree(dA);
        }
        
        cudaFree(U);
        cudaFree(W);
        cudaFree(work);
        cudaFree(hwork);

    }

    //reference implementation in cuSOLVER
    {
        generateUniformMatrix(A,m,n);
        int lwork = 0;
        auto status = cusolverDnSgeqrf_bufferSize(
                ctxt.cusolver_handle, m, n, A, m, &lwork);
        assert(CUSOLVER_STATUS_SUCCESS == status);
        int *devInfo;
        cudaMalloc((void**)&devInfo, sizeof(int));
        float *d_work, *d_tau;
        cudaMalloc((void**)&d_work, sizeof(float)*lwork);
        cudaMalloc((void**)&d_tau, sizeof(float)*m);
        startTimer();
        status = cusolverDnSgeqrf( ctxt.cusolver_handle, m, n, A, m,
                d_tau, d_work, lwork, devInfo);
        assert(CUSOLVER_STATUS_SUCCESS == status);
        float ms = stopTimer();
        printf("CUSOLVER SGEQRF takes %.0f ms, exec rate %.0f GFLOPS\n", ms,
               2.0*n*n*( m -1.0/3.0*n )/(ms*1e6));
        cudaFree(d_work);
        cudaFree(d_tau);
        cudaFree(devInfo);
    }

    cudaFree(A);
    cudaFree(R);

//    cudaFree(R);
//    cudaFree(dA);
    return 0;
}

void checkResult(int m,int n,float* A,int lda, float *Q, int ldq, float *R, int ldr)
{
    float normA = snorm(m,n,A);
    float alpha = 1.0;
    float beta = -1.0;
    startTimer();
    sgemm(m,n,n,Q,ldq,R,ldr,A,lda,alpha,beta);
    float ms = stopTimer();
    printf("SGEMM m*n*k %d*%d*%d takes %.0f (ms), exec rate %.0f GFLOPS\n",
            m, n, n, ms, 2.0*m*n*n/(ms*1e6));
    float normRes = snorm(m,n,A);
    printf("Backward error: ||A-QR||/(||A||) = %.6e\n",normRes/normA);
}

void sgemm(int m,int n,int k,float *dA,int lda, float *dB,int ldb,float *dC, int ldc,float alpha,float beta)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float sone = alpha;
    float szero = beta;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
         m,n,k, 
         &sone, dA, lda, 
         dB, ldb, 
         &szero, dC, ldc
    );
    cublasDestroy(handle);
}

void checkOtho(int m,int n,float *Q, int ldq)
{
    float *I;
    cudaMalloc(&I,sizeof(float)*n*n);

    //printMatrixDeviceBlock("Q.csv",m,n,Q,m);
      
	dim3 grid96( (n+1)/32, (n+1)/32 );
	dim3 block96( 32, 32 );
    setEye<<<grid96,block96>>>( n, n, I, n);
    float snegone = -1.0;
    float sone  = 1.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
        &snegone, Q, CUDA_R_32F, ldq, Q, CUDA_R_32F, ldq,
        &sone, I, CUDA_R_32F, n, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT);
    
    float normRes = snorm(n,n,I);
    printf("||I-Q'*Q||/N = %.6e\n",normRes/n);
    cudaFree(I);
    cublasDestroy(handle);
}


void checkResult(int m,int n, float *A, int lda, float *W, int ldw, float *Y, int ldy , float *R, int ldr)
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

    clearTri<<<grid96,block96>>>('l',n,n,R,ldr);   
    
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
}
