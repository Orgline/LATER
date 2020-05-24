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
// qr left 64
    mgs_caqr_panel_256x32(ctxt, m, 32, A, lda, R, ldr, work);
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        32, 32, m,
        &sone, A, lda,
        &A[32*lda], lda,
        &szero, &R[32*ldr], ldr
    );
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, 32, 32,
        &snegone, A, lda,
        &R[32*ldr], ldr,
        &sone, &A[32*lda], lda
    );
    mgs_caqr_panel_256x32(ctxt, m, 32, &A[32*lda], lda, &R[32+32*ldr], ldr, work);
// update trailing 64
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        64, 64, m,
        &sone, A, lda,
        &A[64*lda], lda,
        &szero, &R[64*ldr], ldr
    );
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, 64, 64,
        &snegone, A, lda,
        &R[64*ldr], ldr,
        &sone, &A[64*lda], lda
    );
    // QR right half 64
    A = &A[64*lda]; 
    R = &R[64*ldr+64];
    mgs_caqr_panel_256x32(ctxt, m, 32, A, lda, R, ldr, work);
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        32, 32, m,
        &sone, A, lda,
        &A[32*lda], lda,
        &szero, &R[32*ldr], ldr
    );
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, 32, 32,
        &snegone, A, lda,
        &R[32*ldr], ldr,
        &sone, &A[32*lda], lda);
    mgs_caqr_panel_256x32(ctxt, m, 32, &A[32*lda], lda, &R[32+32*ldr], ldr, work);
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
    if (n>32) 
    {
        if (threadIdx.x+blockDim.x*blockIdx.x == 0)
            printf("geqrf_tb_256x32: only n<=32 supported. current n=%d\n. Returning.", n);
        return;
    }
    int mm = m - blockIdx.x*256; // TB local number of rows
    mm = (mm<256) ? mm : 256;

    const int mnmin = (mm<n) ? mm : n;

    float *A = &AA[blockIdx.x*256];
    float *R = &RR[blockIdx.x*32];
    //load from global memory to shared memory
    __shared__ float As[256*32], Rs[32*32];
    const int i = threadIdx.x;

    #pragma unroll
    for (int j=0; j<n; j++) 
    {
        if (i<mm) As[i+j*256] = A[i+j*lda];
    }
    __syncthreads();

    const int ldas = 256, ldrs = 32;


    float acc1[1], acc2[1], acc3[1], acc4[1];
    float sum1, sum2, sum3, sum4;

    typedef cub::BlockReduce<float4, 256> BlockReduce;
    typedef cub::BlockReduce<float, 256> BlockReduce2;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ typename BlockReduce2::TempStorage temp_storage2;

    for (int k=0; k<mnmin; k++)
    {
        sum1 = (i<mm) ? As[i+k*ldas]*As[i+k*ldas] : 0;
        float sumsqr = BlockReduce2(temp_storage2).Sum(sum1);

        if (i==0) 
        {
            Rs[k+k*ldrs] = sqrt( sumsqr );
        }
        __syncthreads();

        if (i<mm) 
            As[i+k*ldas] = As[i+k*ldas] / Rs[k+k*ldrs];

        for (int j=k+1; j<(k+4)/4*4 && j<n; j++) 
        {
            sum1 = (i<mm) ? (As[i+k*ldas] * As[i+j*ldas]) : 0;
            float sum = BlockReduce2(temp_storage2).Sum(sum1);
            if (i==0)
                Rs[k+j*ldrs] = sum;
        }

        for (int j=(k+4)/4*4; j<n; j+=4) 
        {
            float4 S;
            S.x = (i<mm) ? (As[i+k*ldas] * As[i+j*ldas]) :    0 ;
            S.y = (i<mm) ? (As[i+k*ldas] * As[i+(j+1)*ldas]): 0 ;
            S.z = (i<mm) ? (As[i+k*ldas] * As[i+(j+2)*ldas]): 0 ;
            S.w = (i<mm) ? (As[i+k*ldas] * As[i+(j+3)*ldas]): 0 ;
            S = BlockReduce(temp_storage).Reduce(S, F4add());
            if (i==0) 
            {
                Rs[k+j*ldrs] = S.x;
                Rs[k+(j+1)*ldrs] = S.y;
                Rs[k+(j+2)*ldrs] = S.z;
                Rs[k+(j+3)*ldrs] = S.w;
            }
        }    

        __syncthreads();

        #pragma unroll
        for (int j=k+1; j<n; j++)
            if (i<mm) 
                As[i+j*ldas] -= As[i+k*ldas]*Rs[k+j*ldrs];

    }
    #pragma unroll
    for (int j=0; j<n; j++) 
    {
        if( i<mm) 
            A[i+j*lda] = As[i+j*ldas];
        if (i<=j) 
        {
            if (i<mm && i<n) 
                R[i+j*ldr] = Rs[i+j*ldrs];
        } 
        else 
        {
            if (i<mm && i<n) 
                R[i+j*ldr] = 0;
        }
    }
}