#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <assert.h>
#include "../include/cub-1.8.0/cub-1.8.0/cub/cub.cuh"
//#include "basicop.cuh"



/*
struct cudaCtxt {
	cublasHandle_t cublas_handle;
	cusolverDnHandle_t cusolver_handle;
};*/

// Communication avoiding QR panel factorization using Householder
// factorization with explicit Q.  
// Templated with fixed number of columns. 
// All pointers are on device. 
// M*N are the block size for CAQR; A reasonable one is 256*16
// On entry, A contains the matrix to be factorized;
// On exit, A contains the explicit Q, R contains the R for which A=QR. 

/*
struct F4add
{ __host__ __device__ __forceinline__
    float4 operator()(const float4& a, const float4& b) const 
    {
    // return a*a;
    float4 c;
    c.x = a.x + b.x; 
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    return c;
    }
};*/

__inline__ __device__ float warpReductionSum(float val) 
{
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float warpAllReduceSum(float val) {
      for (int mask = warpSize/2; mask > 0; mask /= 2) 
          val += __shfl_xor_sync(0xffffffff, val, mask);
      return val;
}

#define TSN 10

#define TIMESTAMP(id) if (threadIdx.x==0)  {\
    long long int c = clock64();\
    if (0) printf("TB[%3d] event[%d] clock[%10lld]\n", blockIdx.x, id, c-timestamps[blockIdx.x]);\
    timestamps[id] = (id==0)? c:  c - timestamps[blockIdx.x];\
}
// must be launched with M threads.  

// Householder CAQR Panel for m*32 or m*16, m = 256*i + 32(16) * j. 
// For m*32 panel ,launch with kernel1<256,32,1024><<< (m+255)/256, {32,32} >>> (...) 
// For m*16 panel ,launch with kernel1<256,16,512><<< (m+255)/256, {32,16} >>> (...) 
// work is not used; just for compatibility.
template<int M, int N, int NT>
__global__ void kernel1( int m, int n, float *AA, int lda, float *RR, int ldr )
{
    printf("Here\n");
    __shared__ long long int timestamps[TSN];
    TIMESTAMP(0)
    int mm = m - blockIdx.x*M; // TB local number of rows
    mm = (mm<M) ? mm : M;
    if (mm <= 0) return; 

    const int mnmin = (mm<n) ? mm : n;

    float *A = &AA[blockIdx.x*M];
    float *R = &RR[blockIdx.x*N];
    __shared__ float As[M*N], Rs[N];
    const int ldas = M, ldrs = N; 

    float acc0, acc1, acc2, acc3, acc4,acc5, acc6, acc7;
    const int i=threadIdx.x, j=threadIdx.y;

#define R07(OP) {OP(0);OP(1);OP(2);OP(3);OP(4);OP(5);OP(6);OP(7);}
#define M1(it) if(threadIdx.x+it*32<mm) As[threadIdx.x+it*32+threadIdx.y*ldas] = A[threadIdx.x+it*32+threadIdx.y*lda]
    //R07(M1);
#pragma unroll
    for (int k=0; k<8; k++) {
        if(i+k*32<mm) As[i+k*32+j*ldas] = A[i+k*32+j*lda];
    }



    __syncthreads();    

    TIMESTAMP(1)
    TIMESTAMP(2)
    for (int k=0; k<mnmin; k++) { 
        // reference: house_gen.m and house_qr from Cleve Moler blog.  
        float nu = 0;
#define M2(it) (threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k)? \
    acc##it = As[threadIdx.x+it*32+threadIdx.y*ldas] * As[threadIdx.x+it*32+threadIdx.y*ldas] : acc##it = 0
#define M2a(it) if (threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k) \
    nu +=  As[threadIdx.x+it*32+threadIdx.y*ldas] * As[threadIdx.x+it*32+threadIdx.y*ldas] 
        if(threadIdx.y==k) {
            if(k==0)TIMESTAMP(3)
            R07(M2)
            nu = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7); 
            if(k==0)TIMESTAMP(4)
            float normxsqr = (warpAllReduceSum(nu));
            float normx = sqrt(normxsqr);
            if(k==0)TIMESTAMP(5)
            //if(threadIdx.x==0 && blockIdx.x==0) 
                //printf("k=%2d, nu=%.3e\n", k, normx );
            float scale = 1.0/normx;
#define M3(it) if(threadIdx.x+it*32<mm&&threadIdx.x+it*32>=k) As[threadIdx.x+it*32+threadIdx.y*ldas] *= scale
            R07(M3);
            if(k==0)TIMESTAMP(6)
            __syncwarp();
            if(threadIdx.x==k) {
                float u1 = As[threadIdx.x+threadIdx.y*ldas];
                //if(blockIdx.x==0) printf("k=%2d, u1=%.3e, [%d,%d]\n", k, u1, threadIdx.x, threadIdx.y);
                As[threadIdx.x+threadIdx.y*ldas] += (u1>=0) ? 1 : -1;
                Rs[k] = (u1>=0)? -normx :normx; 
            }
            __syncwarp();
            scale = 1.0/sqrt(abs(As[k+k*ldas]));
            //if(threadIdx.x==0 && blockIdx.x==0) printf("k=%2d, nu=%.3e, [%d,%d]\n", k, n, threadIdx.x, threadIdx.y);
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

    // Forming explicit Q=Q1*Q2*...*Qn*I. Q is stored in registers Q.
    float Q[8];
#pragma unroll
    for (int k=0; k<8; k++) {
        Q[k] = 0;
    }
    if(i==j) Q[0] = 1.0;
    for (int k=mnmin-1; k>=0; k--) {
        if(threadIdx.y>=k) {
            float acc = 0;
#pragma unroll
            for (int l=0; l<8; l++) 
                acc += As[i+l*32+k*ldas] * Q[l];
            float vq = warpAllReduceSum(acc);
#pragma unroll
            for (int l=0; l<8; l++) 
                if (i+32*l<mm) Q[l] -= vq*( As[i+32*l + k*ldas] );
            
        }
    }
    TIMESTAMP(8)

#pragma unroll
    for (int k=0; k<8; k++) {
        if (i+k*32<mm) A[i+k*32 + j*lda] = Q[k];
    }

    TIMESTAMP(9)
    //if(blockIdx.x==0 && i==0) {
        //for (int j=1; j<TSN; j++) printf("%d\t\t", j); 
        //printf("\n");
        //for (int j=1; j<TSN; j++) {
            //printf("%9lld\t", timestamps[j]); 
        //}
        //printf("\n");
    //}
}

// copied from RGSQRF MGS panel. 
template<int M, int N, int NT>
__global__ void kernel2( int m, int n, float *AA, int lda, float *RR, int ldr, float *work )
{
    __shared__ long long int timestamps[TSN];
    TIMESTAMP(0)
    int mm = m - blockIdx.x*M; // TB local number of rows
    mm = (mm<M) ? mm : M;
    if (mm <= 0) return; 

    const int mnmin = (mm<n) ? mm : n;

    float *A = &AA[blockIdx.x*M];
    float *R = &RR[blockIdx.x*N];
    __shared__ float As[M*N], Rs[N*N];
    const int ldas = M, ldrs = N; 

    const int i = threadIdx.x;
    for (int j=0; j<n; j++) 
    {
#pragma unroll
        if (i<mm)
        	As[i+j*ldas] = A[i+j*lda];
    }
    __syncthreads();    

    TIMESTAMP(1)
    float acc1[1], acc2[1], acc3[1], acc4[1];
    float sum1, sum2, sum3, sum4;

    typedef cub::BlockReduce<float4,NT> BlockReduce;
    typedef cub::BlockReduce<float,NT> BlockReduce2;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ typename BlockReduce2::TempStorage temp_storage2;

    __shared__ float acc[8];
    TIMESTAMP(2)
    for (int k=0; k<mnmin; k++) { 
        sum1 = 0;
        if (i<mm)
            sum1 += As[i+k*ldas]*As[i+k*ldas];
        if(k==0) TIMESTAMP(3)
        float sumsqr = BlockReduce2(temp_storage2).Sum(sum1);
        //float sumsqr = warpreduce(temp_storage3).Sum(sum1);
        //float sumsqr = warpReductionSum(sum1);
        if(k==0) TIMESTAMP(4)

        if (threadIdx.x==0) 
        {
            Rs[k+k*ldrs] = sqrt( sumsqr );
        }
        __syncthreads();
        float ssum = 1.0/Rs[k+k*ldrs];
        if (i<mm)
            As[i+k*ldas] = As[i+k*ldas] * ssum;

        for (int j=k+1; j<(k+4)/4*4 && j<n; j++) 
        {
            sum1 = 0;
            if (i<mm)
                sum1 += As[i+k*ldas] * As[i+j*ldas];
            float sum = BlockReduce2(temp_storage2).Sum(sum1);
            //float sum = warpReductionSum(sum1);
            if (threadIdx.x==0)
                Rs[k+j*ldrs] = sum;
        }
        if(k==0) TIMESTAMP(5);

        for (int j=(k+4)/4*4; j<n; j+=4) 
        {
            float4 S = {0,0,0,0};
            if (i<mm) {
                S.x +=  (As[i+k*ldas] * As[i+j*ldas])     ;
                S.y +=  (As[i+k*ldas] * As[i+(j+1)*ldas]) ;
                S.z +=  (As[i+k*ldas] * As[i+(j+2)*ldas]) ;
                S.w +=  (As[i+k*ldas] * As[i+(j+3)*ldas]) ;
            }
            S = BlockReduce(temp_storage).Reduce(S, F4add());
            //S = warpReductionSum4(S);
            if (threadIdx.x==0) 
            {
                Rs[k+j*ldrs] = S.x;
                Rs[k+(j+1)*ldrs] = S.y;
                Rs[k+(j+2)*ldrs] = S.z;
                Rs[k+(j+3)*ldrs] = S.w;
            }
        }    

        __syncthreads();
        if(k==0) TIMESTAMP(6);

#pragma unroll 4
        for (int j=k+1; j<n; j++)
            if(i<mm)
                As[i+j*ldas] -= As[i+k*ldas]*Rs[k+j*ldrs];
        if(k==0) TIMESTAMP(7);

    }
    TIMESTAMP(8)

#pragma unroll
    for (int j=0; j<n; j++) {
        if( i<mm) 
            A[i+j*lda] = As[i+j*ldas];
        if (i<=j) {
            if (i<mm && i<n) 
                R[i+j*ldr] = Rs[i+j*ldrs];
        } else {
            if (i<mm && i<n) 
                R[i+j*ldr] = 0;
        }
    }    
    TIMESTAMP(9)
    //if(blockIdx.x==0 && threadIdx.x==0) {
        //for (int j=1; j<TSN; j++) printf("%d\t\t", j); 
        //printf("\n");
        //for (int j=1; j<TSN; j++) {
            //printf("%9lld\t", timestamps[j]); 
        //}
        //printf("\n");
    //}
}
// M=256, N=32 or 16. NT=1024 or 512
// work >= 2*m*n floats
template<int M, int N, int NT>
void Panel_CAQR_EXPQ( cudaCtxt ctxt, int m, int n, float *A, int lda, float *R,
        int ldr, float *work)
{
    dim3 blockdim(32, N);
	if ( m <= M ) {
        printMatrixDeviceBlock("AA.csv",m,n,A,lda);
        kernel1<M, N, NT><<<1,blockdim>>>(m, n, A, lda, R, ldr); 
        printMatrixDeviceBlock("QQ.csv",m,n,A,lda);
        printMatrixDeviceBlock("RR.csv",n,n,R,ldr);
		return; 
	}

    if ( (m-m/M*M)%N != 0) {
        printf("Error: m must be i*%d + j*%d\n", M, N);
    }
    int NB = (m+M-1)/M;
	int ldwork = NB*N; 
    int mm = NB*N; 
    //printf("NB = %d, ldwork = %d, mm = %d, m = %d, n = %d, lda = %d, ldr = %d, M = %d, N = %d\n",NB,ldwork,mm,m,n,lda,ldr,M,N);
    //printMatrixDeviceBlock("A.csv",m,n,A,lda);
    kernel1<M,N,NT><<<NB,blockdim>>>(m, n, A, lda, work, ldwork); 
    //printMatrixDeviceBlock("QQ.csv",m,n,A,lda);
    //printMatrixDeviceBlock("RR.csv",mm,n,work,ldwork);
    Panel_CAQR_EXPQ<M,N,NT>( ctxt, mm, n, work, ldwork, R, ldr,  work+ldwork*n );
    //printMatrixDeviceBlock("QQQ.csv",mm,n,work,ldwork);
    //printMatrixDeviceBlock("RRR.csv",n,n,R,ldr);
    
	float sone = 1.0, szero = 0.0;
    cublasSgemmStridedBatched(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, N,
        &sone, A, lda, M,
        work, ldwork, N,
        &szero, A,lda, M,
        m/M);
    mm = m%M;
    if (mm>0) {
        cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                mm, N, N, &sone, &A[m/M*M], lda, &work[m/M*N], ldwork, 
                &szero, &A[m/M*M], lda);
    }
    /*
    dim3 grid( (m+31)/32, (n+31)/32 );
    dim3 block( 32, 32 );
    myslacpy<<<grid, block>>>( m, n, work+ldwork*N, lda, A,  lda );    */	
    //printMatrixDeviceBlock("Q.csv",m,n,A,lda);
    //printMatrixDeviceBlock("R.csv",n,n,R,ldr);
}