#include "LATER.h"
#include "LATER_QR.h"
#include "../include/cub-1.8.0/cub-1.8.0/cub/cub.cuh"

void mgs_caqr_panel_256x128(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work)
{
    if (m<256 || n!=128) 
    {
        printf("CAQR_256x128: ERROR: m must be > 256, n must be 128. (m,n)=(%d,%d)\n", m, n);
    }
    float sone = 1.0, szero = 0.0, snegone = -1.0;

}

void mgs_caqr_panel_256x32(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work)
{
    if (n!=32) 
    {
        printf("[Error]: CAQR_32 does not support n!=32\n");
        return;
    }


    if (m <= 256) 
    {
        // printf("CAQR: Recursion tree leaf: ");
        mgs_kernel<<<1,256>>>(m, n, A,  lda, R, ldr);
    } 
    else 
    { // m > 256, recurse.
        
        float sone = 1.0, szero = 0.0;
        if (m%256 == 0) 
        {
            int ldwork = m/256*32;
            int mm = m/256*32;

            mgs_kernel<<<m/256,256>>>(m, n, A,  lda, work, ldwork);

            mgs_caqr_panel_256x32(ctxt, mm , n, work, ldwork, R, ldr, work+ldwork*n);

            
            cublasSgemmStridedBatched(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            256, 32, 32,
            &sone, A, lda, 256,
            work, ldwork, 32,
            &szero, A,lda, 256,
            m/256);
            
        }
        else
        {
            int nb = m/256;
            int r = m%256;
            int ldwork = m/256*32+32;
            int mm = m/256*32+32;
            //printMatrixDeviceBlock("A.csv",m,n,A,lda);
            mgs_kernel<<<m/256+1,256>>>(m, n, A,  lda, work, ldwork);
            mgs_caqr_panel_256x32(ctxt, mm , n, work, ldwork, R, ldr, work+ldwork*n);
            cublasSgemmStridedBatched(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                256, 32, 32,
                &sone, A, lda, 256,
                work, ldwork, 32,
                &szero, A,lda, 256,
                nb
            );
            
            cublasSgemm(ctxt.cublas_handle,CUBLAS_OP_N, CUBLAS_OP_N,
                r, 32,32,
                &sone,A+nb*256,lda,
                work+nb*32, ldwork,
                &szero, A+nb*256,lda
            );
            //CAQR_256x32(ctxt, mm , n, work, ldwork, R, ldr, work+ldwork*n);
        }
    }
}

__global__ void mgs_kernel(int m, int n, float *AA, int lda, float *RR, int ldr)
{

}