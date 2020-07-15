#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <curand.h>
#include <stdlib.h>
#include <assert.h>
#include "LATER.h"

#ifdef MAGMA
#include "magma_v2.h"
#endif

int main(int argc, char *argv[]) 
{
#ifdef MAGMA
    printf("MAGMA LIB FOUND!\n");

#endif
    if (argc < 2) {
        printf(" usage: %s <n> \n", argv[0]); 
        return 0; 
    }
    int n = atoi(argv[1]); 


    cusolverDnHandle_t cusolver_handle = NULL; 
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; 

    cusolver_status = cusolverDnCreate(&cusolver_handle); 
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status); 

    float *A, *D, *E, *tau; 
    //int n = 8192; 
    int lda = n; 
    cudaMalloc( &A, sizeof(*A) * n * lda ); 
    cudaMalloc( &D, sizeof(*A) * n  ); 
    cudaMalloc( &E, sizeof(*A) * n  ); 
    cudaMalloc( &tau, sizeof(*A) * n  ); 

    curandGenerator_t gen; 
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); 

    {
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); 
        curandGenerateUniform(gen, A, n*lda); 

        int lwork;
        cusolverDnSsytrd_bufferSize( cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, A, lda, D, E, tau,  &lwork);
        printf("n=%d, Ssytrd buffsize lwork=%d\n", n, lwork); 

        float *work;
        int *devInfo; 
        cudaMalloc( &work, sizeof(*work)*lwork );
        cudaMalloc( &devInfo, sizeof(*devInfo));
        startTimer();
        cusolverDnSsytrd( cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, A, lda, D, E, tau, work, lwork, devInfo);
        float ms = stopTimer(); // in milliseconds
        int info; 
        cudaMemcpy( &info, devInfo, sizeof(info), cudaMemcpyDeviceToHost );
        float GFLOPS = 4.0/3.0 * n * n * n / (ms*1.0e6);
        printf(" Ssytrd info=%d took time %.0f milliseconds GFLOPS: %.0f\n", info, ms, GFLOPS );
    }
    {
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); 
        curandGenerateUniform(gen, A, n*lda); 

        int lwork;
        auto jobz = CUSOLVER_EIG_MODE_NOVECTOR; 
        //auto jobz = CUSOLVER_EIG_MODE_VECTOR; 
        cusolverDnSsyevd_bufferSize( cusolver_handle, jobz, CUBLAS_FILL_MODE_LOWER, n, A, lda, D,  &lwork);
        printf("n=%d, Ssyevd buffsize lwork=%d\n", n, lwork); 

        float *work;
        int *devInfo; 
        cudaMalloc( &work, sizeof(*work)*lwork );
        cudaMalloc( &devInfo, sizeof(*devInfo));
        startTimer();
        cusolverDnSsyevd( cusolver_handle, jobz, CUBLAS_FILL_MODE_LOWER, n, A, lda, D, work, lwork, devInfo);
        float ms = stopTimer(); // in milliseconds
        int info; 
        cudaMemcpy( &info, devInfo, sizeof(info), cudaMemcpyDeviceToHost );
        float GFLOPS = 4.0/3.0 * n * n * n / (ms*1.0e6);
        printf(" Ssyevd (novector) info=%d took time %.0f milliseconds GFLOPS: %.0f\n", info, ms, GFLOPS );
    }
    {
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); 
        curandGenerateUniform(gen, A, n*lda); 

        int lwork;
        auto jobz = CUSOLVER_EIG_MODE_VECTOR; 
        //auto jobz = CUSOLVER_EIG_MODE_VECTOR; 
        cusolverDnSsyevd_bufferSize( cusolver_handle, jobz, CUBLAS_FILL_MODE_LOWER, n, A, lda, D,  &lwork);
        printf("n=%d, Ssyevd buffsize lwork=%d\n", n, lwork); 

        float *work;
        int *devInfo; 
        cudaMalloc( &work, sizeof(*work)*lwork );
        cudaMalloc( &devInfo, sizeof(*devInfo));
        startTimer();
        cusolverDnSsyevd( cusolver_handle, jobz, CUBLAS_FILL_MODE_LOWER, n, A, lda, D, work, lwork, devInfo);
        float ms = stopTimer(); // in milliseconds
        int info; 
        cudaMemcpy( &info, devInfo, sizeof(info), cudaMemcpyDeviceToHost );
        float GFLOPS = 4.0/3.0 * n * n * n / (ms*1.0e6);
        printf(" Ssyevd (with vector) info=%d took time %.0f milliseconds GFLOPS: %.0f\n", info, ms, GFLOPS );
    }
#ifdef MAGMA
    {
        magma_print_environment();
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); 
        curandGenerateUniform(gen, A, n*lda); 
        magma_init();
        printf("hey magma worked! let's do some number crunching\n"); 
        int nb = 128; 
        int lwork = 2*n*nb; 
        int info; 

        float *hA = (float*) malloc( sizeof(*hA) * n * lda ); 
        cudaMemcpy( hA, A, sizeof(*hA)*n*lda, cudaMemcpyDeviceToHost ); 
        magma_ssytrd_sy2sb(MagmaLower, n, nb, A, lda, tau, work, lwork, &info); 
        magma_finalize();
    }
#endif
}
