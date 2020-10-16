#include "LATER.h"
#include "OC_gemm.h"

#define BLOCKSIZE 2048

/*
Output:
A stores the Q on host
R stores the R on host
*/

float oc_panel_time = 0.0;
float oc_gemm_time = 0.0;
void panel(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float* dA, float *dB, float *dR, __half *hwork)
{
    //startTimer();
    cudaMemcpy(dA, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    int lwork = m / 256 * 32 * n;
    int lhwork = m * n;
    
    later_rgsqrf(ctxt, m, n, dA, lda, dR, n, dB, lwork, hwork, lhwork);
    
    cublasGetMatrix(m, n, sizeof(float), dA, lda, A, lda);
    cublasGetMatrix(n, n, sizeof(float), dR, n, R, ldr);
    //float ms = stopTimer();
    //printf("%fms\n",ms);
}

void later_oc_qr_rec(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, std::shared_ptr<Mem_pool> pool)
{
    if(n <= BLOCKSIZE)
    {
        startTimer();
        std::cout << "pool free " << pool->size() << std::endl;
        float *dA = reinterpret_cast<float *>(pool->allocate(sizeof(float)*m*n));
        std::cout << "pool free " << pool->size() << std::endl;
        float *dB = reinterpret_cast<float *>(pool->allocate(sizeof(float)*m/256*32*n));
        std::cout << "pool free " << pool->size() << std::endl;
        float *dR = reinterpret_cast<float *>(pool->allocate(sizeof(float)*n*n));
        std::cout << "pool free " << pool->size() << std::endl;
        /*
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        std::cout << free << "\t" << total << std::endl;
        */
        __half *hwork = reinterpret_cast<__half *>(pool->allocate(sizeof(__half)*m*n));
        std::cout << "pool free " << pool->size() << std::endl;
        //startTimer();
        panel(ctxt, m, n, A, lda, R, ldr, dA, dB, dR, hwork);

        //printf("m,n = %d,%d\n", m,n);
        //
        pool->free(hwork);
        pool->free(dR);
        pool->free(dB);
        pool->free(dA);
        float ms = stopTimer();
        oc_panel_time += ms;
        printf("Panel takes %lf ms\n", oc_panel_time);
        return;
    }
    //left recurse
    later_oc_qr_rec(ctxt,m, n / 2, A, lda, R, ldr, pool);

    float sone = 1.0;
    float szero = 0.0;
    float snegone = -1.0;


    startTimer();
    OC_gemm OC(n/2, n/2, m, pool);

    OC.gemm(CUBLAS_OP_T, CUBLAS_OP_N, sone, A, lda, A+n/2*lda, lda, szero, R+n/2*ldr, ldr);

    //delete &OC1;

    //OC_gemm OC2(m, n/2, n/2);

    OC.gemm(CUBLAS_OP_N, CUBLAS_OP_N, snegone, A, lda,R+n/2*ldr, ldr, sone, A+n/2*lda, lda);

    float ms = stopTimer();
    oc_gemm_time += ms;
    printf("Gemm takes %lf ms\n", oc_gemm_time);

    //delete &OC;

    //delete &OC2;

    //later_oc_sgemm(CUBLAS_OP_T, CUBLAS_OP_N, n/2, n/2, m, sone, A, lda, A+n/2*lda, lda, szero, R+n/2*ldr, ldr);

    //later_oc_sgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n/2, n/2, snegone, A, lda, R+n/2*ldr, ldr, sone, A+n/2*lda, lda);

    later_oc_qr_rec(ctxt, m, n/2, A+n/2*lda, lda, R+n/2+n/2*ldr, ldr, pool);
}

void later_oc_qr_blk(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, std::shared_ptr<Mem_pool> pool)
{
    float sone = 1.0;
    float szero = 0.0;
    float snegone = -1.0;
    for(int i = 0; i < n; i += BLOCKSIZE)
    {
        float *dA = reinterpret_cast<float *>(pool->allocate(sizeof(float)*m*BLOCKSIZE));
        float *dB = reinterpret_cast<float *>(pool->allocate(sizeof(float)*m/256*32*BLOCKSIZE));
        float *dR = reinterpret_cast<float *>(pool->allocate(sizeof(float)*BLOCKSIZE*BLOCKSIZE));
        __half *hwork = reinterpret_cast<__half *>(pool->allocate(sizeof(__half)*m*BLOCKSIZE));
        panel(ctxt, m, n, A+i*lda, lda, R+i+i*ldr, ldr, dA, dB, dR, hwork);
        pool->free(dA);
        pool->free(dB);
        pool->free(dR);
        pool->free(hwork);

        if(i < n - BLOCKSIZE)
        {
            //later_oc_sgemm(CUBLAS_OP_T, CUBLAS_OP_N, BLOCKSIZE, n-i-BLOCKSIZE, m, sone, A+i*lda, lda, A+(i+BLOCKSIZE)*lda, lda, szero, R+i+BLOCKSIZE*ldr, ldr);

            //later_oc_sgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n-i-BLOCKSIZE, BLOCKSIZE , snegone, A+i*lda, lda, R+i+BLOCKSIZE*ldr, ldr, sone, A+(i+BLOCKSIZE)*lda, lda);
        }
    }
}

