#include "LATER.h"
#include "LATER_QR.h"
#include <assert.h>

#define NMIN 128

int algo;
int m,n;

void checkResult(int m,int n,float* A,int lda, float *Q, int ldq, float *R, int ldr);
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
        printf("\t-check: enable checking the orthogonality and backward error\n");
    }
    if(parseArguments(argc,argv)!=0)
    {
        return 0;
    }
    float *A;
    cudaMalloc(&A,sizeof(float)*m*n);
    float *R;
    cudaMalloc(&R,sizeof(float)*n*n);

    generateUniformMatrix(A,m,n);

    float *dA;
//    cudaMalloc(&dA,sizeof(float)*m*n);
//    cudaMemcpy(dA,A,sizeof(float)*m*n,cudaMemcpyDeviceToDevice);

    cudaCtxt ctxt {};
    cublasCreate(&ctxt.cublas_handle );
    cusolverDnCreate(&ctxt.cusolver_handle );

    int lwork;

    cusolverDnSgeqrf_bufferSize(
        ctxt.cusolver_handle,
        m,
        NMIN,
        A,
        m,
        &lwork
    );
    float *work;
    cudaMalloc( &work, lwork * sizeof(float) );
    
    __half *hwork;
    int lhwork = m*n;
    cudaMalloc( &hwork, sizeof(__half) * lhwork );

        
    if (algo == 1)
    {
        printf("Perform RGSQRF\nmatrix size %d*%d\n",m,n);
        startTimer();
        later_rgsqrf(m,n,A,m,R,n,work,lwork,hwork,lhwork);
        float ms = stopTimer();
        printf("RGSQRF takes %.0f ms, exec rate %.0f GFLOPS\n", ms, 
                2.0*n*n*( m -1.0/3.0*n )/(ms*1e6));

        if (checkFlag) {
            cudaMalloc(&dA,sizeof(float)*m*n);
            generateUniformMatrix(dA,m,n);
            printf("Orthogonality ");
            checkOtho(m, n, A, m);

            printf("Backward error ");
            checkResult(m, n, dA, m, A, m, R, n);
            cudaFree(dA);
        }
        cudaFree(R);
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
        free(d_work);
    }

    cudaFree(A);
//    cudaFree(R);
//    cudaFree(dA);
    return 0;
}

void checkResult(int m,int n,float* A,int lda, float *Q, int ldq, float *R, int ldr)
{
    float normA = snorm(m,n,A);
    float alpha = 1.0;
    float beta = -1.0;
    sgemm(m,n,n,Q,ldq,R,ldr,A,lda,alpha,beta);
    float normRes = snorm(m,n,A);
    printf("||A-QR||/(||A||) = %.6e\n",normRes/normA);
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
