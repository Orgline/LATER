#include "LATER.h"
#include "LATER_QR.h"
#include "../include/cub-1.8.0/cub-1.8.0/cub/cub.cuh"
#include <assert.h>
/*
This panel serves later_rgsqrf
*/


void mgs_caqr_panel_256x128(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work)
{
    if (m<256 || n!=128) 
    {
        printf("CAQR_256x128: ERROR: m must be > 256, n must be 128. (m,n)=(%d,%d)\n", m, n);
    }
    float sone = 1.0;
    float szero = 0.0;
    float snegone= -1.0;
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
        
        float sone = 1.0;
        float szero = 0.0;
        
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


    //float acc1[1], acc2[1], acc3[1], acc4[1];
    float sum1;

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

/*
 * A faster panel for rgsqrf().
 */
__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}
__global__ void mgs_kernel2(int m, int n, float *AA, int lda, float *RR, int ldr)
{

    int mm = m - blockIdx.x*256; // TB local number of rows
    mm = (mm<256) ? mm : 256;

    const int mnmin = (mm<n) ? mm : n;

    //float *A = &AA[blockIdx.x*256];
    //float *R = &RR[blockIdx.x*32];
    //load from global memory to shared memory
    __shared__ float As[256], Rs[32*32];

#define ldrs  32
    //if(i==0 && j==0) printf("hello!\n");
    float Ar[8]; // register files

    // load block A into registers.
#pragma unroll 4
    for (int l = 0; l < 8; l++) {
        if (threadIdx.x + l * 32 < mm && threadIdx.y < mnmin) {
            Ar[l] = AA[blockIdx.x * 256 + threadIdx.x + l * 32 + threadIdx.y * lda];
        }
    }

    __syncthreads();

    for (int k=0; k<mnmin; k++) {
        float nu = 0; // acc for norm

        if (threadIdx.y==k) {
#pragma unroll 8
            for(int l=0; l<8; l++) {
                nu +=  (threadIdx.x+l*32<mm) ? (Ar[l]*Ar[l]) : 0;
            }
            float normx = sqrt((warpAllReduceSum(nu)));
            if(threadIdx.x==k) {
                Rs[k+k*ldrs] = normx;
            }
            float scale = 1.0f/normx;
#pragma unroll 8
            for(int l=0; l<8; l++) {
                if(threadIdx.x+l*32<mm) {
                    Ar[l] *= scale;
                    As[threadIdx.x+l*32] = Ar[l];
                }
            }
        }
        __syncthreads();
        nu = 0;
        if (threadIdx.y>k) {
#pragma unroll 8
            for (int l = 0; l < 8; l++) {
                if (threadIdx.x+l*32<mm) {
                    nu += (As[threadIdx.x + l * 32] * Ar[l]);
                }
            }
            float scale = (warpAllReduceSum(nu));
#pragma unroll 8
            for (int l=0; l<8; l++) {
                if (threadIdx.x+l*32<mm) {
                    Ar[l] -= As[threadIdx.x+l*32] * scale;
                }
            }
            if (threadIdx.x==k) Rs[k+threadIdx.y*ldrs] = scale;
        }
        __syncthreads();
    }


#pragma unroll 8
    for (int l = 0; l < 8; l++) {
        if (threadIdx.x + l * 32 < mm && threadIdx.y < mnmin)
            AA[blockIdx.x * 256 + threadIdx.x + l * 32 + threadIdx.y * lda] = Ar[l];
    }
    if (threadIdx.x < mnmin && threadIdx.y < mnmin)
        RR[blockIdx.x * 32 + threadIdx.x + threadIdx.y * ldr] =
                (threadIdx.x <= threadIdx.y) ? Rs[threadIdx.x +threadIdx.y * ldrs]: 0;

}

/*
This part serves later_rhouqr
*/

__inline__ __device__ float warpReductionSum(float val)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}




template<int M, int N>
void hou_caqr_panel( cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work)
{
    dbgprintf("hou_caqr_panel(m=%d,n=%d,lda=%d,ldr=%d",m, n, lda, ldr);
    dim3 blockdim(32, N);
    if ( m <= M ) {
        dbgprintf("launching hou_kernel<%d,%d><<<(%d),(%d,%d)>>>(%d,%d,%d,%d)\n",
                M, N, 1, blockdim.x, blockdim.y, m, n, lda, ldr);
        hou_kernel<M, N><<<1,blockdim>>>(m, n, A, lda, R, ldr);

        CHECK_KERNEL();

        return;
    }
    if ( (m-m/M*M)%N != 0) {
        printf("Error: m must be i*%d + j*%d\n", M, N);
    }
    int NB = (m+M-1)/M;
    int ldwork = NB*N;
    int mm = NB*N;
    dbgprintf("launching hou_kernel<%d,%d><<<(%d),(%d,%d)>>>(%d,%d,%d,%d)\n",
           M, N, NB, blockdim.x, blockdim.y, m, n, lda, ldwork);
    hou_kernel<M,N><<<NB,blockdim>>>(m, n, A, lda, work, ldwork);

    CHECK_KERNEL();
    hou_caqr_panel<M,N>( ctxt, mm, n, work, ldwork, R, ldr,  work+ldwork*n );
    float sone = 1.0, szero = 0.0;
    auto status = cublasSgemmStridedBatched(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, N,
                              &sone, A, lda, M,
                              work, ldwork, N,
                              &szero, A,lda, M,
                              m/M);
    assert(CUBLAS_STATUS_SUCCESS == status);
    
    mm = m%M;
    if (mm>0) {
        auto status = cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    mm, N, N, &sone, &A[m/M*M], lda, &work[m/M*N], ldwork,
                    &szero, &A[m/M*M], lda);
        assert(CUBLAS_STATUS_SUCCESS == status);
    }

}
template<int M, int N>
__global__ void hou_kernel2( int m, int n, float *AA, int lda, float *RR, int ldr )
{

    int mm = m - blockIdx.x*M; // TB local number of rows
    mm = (mm<M) ? mm : M;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        dbgprintf("bid[%d] mm=%d\n", blockIdx.x, mm);
    if (mm <= 0) return;

    const int mnmin = (mm<n) ? mm : n;

    float *A = &AA[blockIdx.x*M];
    float *R = &RR[blockIdx.x*N];
    __shared__ float As[M*N], Rs[N];
    const int ldas = M/*, ldrs = N*/;

    float acc0, acc1, acc2, acc3, acc4,acc5, acc6, acc7;
    const int i=threadIdx.x, j=threadIdx.y;

#define R07(OP) {OP(0);OP(1);OP(2);OP(3);OP(4);OP(5);OP(6);OP(7);}
#define M1(it) if(threadIdx.x+it*32<mm) As[threadIdx.x+it*32+threadIdx.y*ldas] = A[threadIdx.x+it*32+threadIdx.y*lda]

#pragma unroll 4
    for (int k=0; k<8; k++) {
        if(i+k*32<mm) As[i+k*32+j*ldas] = A[i+k*32+j*lda];
    }

    __syncthreads();

    for (int k=0; k<mnmin; k++) {
        // reference: house_gen.m and house_qr from Cleve Moler blog.
        float nu = 0;
#define M2(it) (threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k)? \
    acc##it = As[threadIdx.x+it*32+threadIdx.y*ldas] * As[threadIdx.x+it*32+threadIdx.y*ldas] : acc##it = 0
#define M2a(it) if (threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k) \
    nu +=  As[threadIdx.x+it*32+threadIdx.y*ldas] * As[threadIdx.x+it*32+threadIdx.y*ldas]
        if(threadIdx.y==k) {

            R07(M2)
            nu = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);

            float normxsqr = (warpAllReduceSum(nu));
            float normx = sqrt(normxsqr);

            float scale = 1.0/normx;
#define M3(it) if(threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k) As[threadIdx.x+it*32+threadIdx.y*ldas] *= scale
            R07(M3);

            __syncwarp();
            if(threadIdx.x==k) {
                float u1 = As[threadIdx.x+threadIdx.y*ldas];

                As[threadIdx.x+threadIdx.y*ldas] += (u1>=0) ? 1 : -1;
                Rs[k] = (u1>=0)? -normx :normx;
            }
            __syncwarp();
            scale = 1.0/sqrt(abs(As[k+k*ldas]));

            R07(M3);
            __syncwarp();
        }
        __syncthreads();
        if(threadIdx.y>k) {
            float uxl = 0;
#define M4(it) (threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k)? \
    acc##it = As[threadIdx.x+it*32+threadIdx.y*ldas] * As[threadIdx.x+it*32+k*ldas]: acc##it = 0;
#define M4a(it) if(threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k) \
    uxl += As[threadIdx.x+it*32+threadIdx.y*ldas] * As[threadIdx.x+it*32+k*ldas]
            R07(M4)
            uxl = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);
            float ux = warpAllReduceSum(uxl);
#define M5(it) if(threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k) \
    As[threadIdx.x+it*32+threadIdx.y*ldas] -= ux * As[threadIdx.x+it*32+k*ldas]
            R07(M5)
        }
    }

    __syncthreads();
    if (i==j) R[i+j*ldr] = Rs[i];
    else if (i<j) {
        R[i+j*ldr] = As[i+j*ldas];
        As[i+j*ldas] = 0;
    }
    else if (i<n) {
        R[i+j*ldr] = 0;
    }

    float Q[8];
#pragma unroll 4
    for (int k=0; k<8; k++) {
        Q[k] = 0;
    }
    if(i==j) Q[0] = 1.0;
    for (int k=mnmin-1; k>=0; k--) {
        if(threadIdx.y>=k) {
            float acc = 0;
#pragma unroll 4
            for (int l=0; l<8; l++)
                acc += As[i+l*32+k*ldas] * Q[l];
            float vq = warpAllReduceSum(acc);
#pragma unroll 4
            for (int l=0; l<8; l++)
                if (i+32*l<mm) Q[l] -= vq*( As[i+32*l + k*ldas] );

        }
    }

#pragma unroll 4
    for (int k=0; k<8; k++) {
        if (i+k*32<mm) A[i+k*32 + j*lda] = Q[k];
    }


}
template<int M, int N>
__global__ void hou_kernel( int m, int n, float *AA, int lda, float *RR, int ldr )
{

    int mm = m - blockIdx.x*M; // TB local number of rows
    mm = (mm<M) ? mm : M;

    if (mm <= 0) return;

    const int mnmin = (mm<n) ? mm : n;

    float *A = &AA[blockIdx.x*M];
    float *R = &RR[blockIdx.x*N];
    __shared__ float As[M*N], Rs[N];
    const int ldas = M/*, ldrs = N*/;

    float acc0, acc1, acc2, acc3, acc4,acc5, acc6, acc7;
    const int i=threadIdx.x, j=threadIdx.y;

#define R07(OP) {OP(0);OP(1);OP(2);OP(3);OP(4);OP(5);OP(6);OP(7);}
#define M1(it) if(threadIdx.x+it*32<mm) As[threadIdx.x+it*32+threadIdx.y*ldas] = A[threadIdx.x+it*32+threadIdx.y*lda]

#pragma unroll 4
    for (int k=0; k<8; k++) {
        if(i+k*32<mm) As[i+k*32+j*ldas] = A[i+k*32+j*lda];
    }

    __syncthreads();

    for (int k=0; k<mnmin; k++) {
        // reference: house_gen.m and house_qr from Cleve Moler blog.
        float nu = 0;
#define M2(it) (threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k)? \
    acc##it = As[threadIdx.x+it*32+threadIdx.y*ldas] * As[threadIdx.x+it*32+threadIdx.y*ldas] : acc##it = 0
#define M2a(it) if (threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k) \
    nu +=  As[threadIdx.x+it*32+threadIdx.y*ldas] * As[threadIdx.x+it*32+threadIdx.y*ldas]
        if(threadIdx.y==k) {

            R07(M2)
            nu = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);

            float normxsqr = (warpAllReduceSum(nu));
            float normx = sqrt(normxsqr);

            float scale = 1.0/normx;
#define M3(it) if(threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k) As[threadIdx.x+it*32+threadIdx.y*ldas] *= scale
            R07(M3);

            __syncwarp();
            if(threadIdx.x==k) {
                float u1 = As[threadIdx.x+threadIdx.y*ldas];

                As[threadIdx.x+threadIdx.y*ldas] += (u1>=0) ? 1 : -1;
                Rs[k] = (u1>=0)? -normx :normx;
            }
            __syncwarp();
            scale = 1.0/sqrt(abs(As[k+k*ldas]));

            R07(M3);
            __syncwarp();
        }
        __syncthreads();
        if(threadIdx.y>k) {
            float uxl = 0;
#define M4(it) (threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k)? \
    acc##it = As[threadIdx.x+it*32+threadIdx.y*ldas] * As[threadIdx.x+it*32+k*ldas]: acc##it = 0;
#define M4a(it) if(threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k) \
    uxl += As[threadIdx.x+it*32+threadIdx.y*ldas] * As[threadIdx.x+it*32+k*ldas]
            R07(M4)
            uxl = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);
            float ux = warpAllReduceSum(uxl);
#define M5(it) if(threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k) \
    As[threadIdx.x+it*32+threadIdx.y*ldas] -= ux * As[threadIdx.x+it*32+k*ldas]
            R07(M5)
        }
    }

    __syncthreads();
    if (i==j) R[i+j*ldr] = Rs[i];
    else if (i<j) {
        R[i+j*ldr] = As[i+j*ldas];
        As[i+j*ldas] = 0;
    }
    else if (i<n) {
        R[i+j*ldr] = 0;
    }

    float Q[8];
#pragma unroll 4
    for (int k=0; k<8; k++) {
        Q[k] = 0;
    }
    if(i==j) Q[0] = 1.0;
    for (int k=mnmin-1; k>=0; k--) {
        if(threadIdx.y>=k) {
            float acc = 0;
#pragma unroll 4
            for (int l=0; l<8; l++)
                acc += As[i+l*32+k*ldas] * Q[l];
            float vq = warpAllReduceSum(acc);
#pragma unroll 4
            for (int l=0; l<8; l++)
                if (i+32*l<mm) Q[l] -= vq*( As[i+32*l + k*ldas] );

        }
    }

#pragma unroll 4
    for (int k=0; k<8; k++) {
        if (i+k*32<mm) A[i+k*32 + j*lda] = Q[k];
    }


}


template void hou_caqr_panel<256,32>( cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work);
template __global__ void hou_kernel2<256,32>( int m, int n, float *AA, int lda, float *RR, int ldr );