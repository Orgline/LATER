#include <cuda.h>
#include <cooperative_groups.h>
#include "LATER.h"
#include "LATER_QR.h"
#include <cassert>

using namespace cooperative_groups;

#define FIRST_IN_WARP if((threadIdx.x + blockDim.x*threadIdx.y)%32==0)
#define FIRST_IN_TB   if(threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0)
#define FIRST_IN_GRID if(blockIdx.x==0 && threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0)

__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// intra-threadblock inner product
// nn is local number of elements.
// nb is the block size;
// X is global accumulator; need to be set to 0 on entry;
// computes a[0:n].*b[0:n].
 __device__ float dot_prod(float *a, float *b, int m, int mb, float *X)
{
    grid_group grid;

    FIRST_IN_GRID *X = 0;
    grid.sync();

    float *aa = &a[blockIdx.x * mb];
    float *bb = &b[blockIdx.x * mb];
    int nn = min(m - blockIdx.x * mb, mb);
    __shared__ float acc[32];
    if (threadIdx.x==0) acc[threadIdx.y] = 0;
    __syncthreads();

    int bxy = blockDim.x*blockDim.y;
    // use all threads to do this.
    for (int i=0; i<(nn+bxy-1)/bxy; i++) {
        int ii = i*bxy + threadIdx.y*32 + threadIdx.x;
        float val = 0, nu = 0;
        if (ii < nn) {
            val = aa[ii] * bb[ii];
        }
        nu = warpAllReduceSum(val);
        __syncwarp();
        if (threadIdx.x == 0)
            atomicAdd(&acc[threadIdx.y], nu);
        __syncwarp();
    }
    __syncthreads();
    if (threadIdx.y == 0) { // first warp collects.
        float val = (threadIdx.x < (mb+31) / 32) ?
                acc[threadIdx.x] : 0;

        float nu = warpAllReduceSum(val);
        __syncwarp(); // NOTE: This is important!
        if (threadIdx.x==0) {
            atomicAdd(X, nu);
        }
    }
    grid.sync();
    return *X;
}

// y = y + a*x;
__device__ void saxpy(float *x, float *y, float a, int n, int nb)
{
    grid_group grid;
    float *xx = &x[blockIdx.x*nb];
    float *yy = &y[blockIdx.x*nb];

    int nn = min(nb, n - blockIdx.x * nb);
    int bxy = blockDim.x * blockDim.y;
    for (int i=0; i<(nn+bxy-1)/bxy; i++) {
        int ii = i*bxy + threadIdx.y*32 + threadIdx.x;
        if (ii < nn) {
            yy[ii] += a*xx[ii];
        }
    }
    grid.sync();
}

/* Modified Gram Schmidt
    function [Q,R] =  mgs(X)
        % Modified Gram-Schmidt.  [Q,R] = mgs(X);
        % G. W. Stewart, "Matrix Algorithms, Volume 1", SIAM, 1998.
        [n,p] = size(X);
        Q = zeros(n,p);
        R = zeros(p,p);
        for k = 1:p
            Q(:,k) = X(:,k);
            for i = 1:k-1
                R(i,k) = Q(:,i)'*Q(:,k);
                Q(:,k) = Q(:,k) - R(i,k)*Q(:,i);
            end
            R(k,k) = norm(Q(:,k))';
            Q(:,k) = Q(:,k)/R(k,k);
        end
    end


 */
__global__ void mgs_panel_general_kernel(int m, int n, float *A, int lda, float *R, int ldr, int mb, float *work)
{
    grid_group grid;
//    if(threadIdx.x==0&&threadIdx.x==0&&threadIdx.y==0) printf("mgs_general, m, n=%d\n", m, n);
//    int ii = blockIdx.x*mb + threadIdx.y*32 + threadIdx.x;
//    int bxy = blockDim.x * blockDim.y;
    for (int k=0; k<n; k++) {
        for (int i=0; i<k; i++) {
            float Rik = dot_prod(&A[lda*i], &A[lda*k], m, mb, &work[i]);
            FIRST_IN_GRID {
                R[i + ldr*k] = Rik;
            }

            saxpy(&A[lda*i], &A[lda*k], -Rik, m, mb);

        }
        float Rkk = sqrt(dot_prod(&A[lda*k], &A[lda*k], m, mb, &work[k]));
//        FIRST_IN_GRID
//            printf("[k=%2d] R[k,k]=%f\n", k, Rkk);

        float s = 1/Rkk;
        {
            int mm = min(m - blockIdx.x*mb, mb);
            int ii = threadIdx.y*32 + threadIdx.x;
            if (ii<mm)
                A[lda*k + blockIdx.x*mb + ii] *= s;
        }

        if (blockIdx.x==0 && threadIdx.x==0 && threadIdx.y==0) {
            R[k + ldr*k] = Rkk;
        }
//        grid.sync(); // This is important!
    }
}


// general mgs QR. Can be used as panel for cleanup residual factorization.
// no restriction on m, and n.
// work size >= n;
void mgs_panel_general(int m, int n, float *A, int lda, float *R, int ldr, float *work)
{
    int mp;
    cudaDeviceGetAttribute(&mp, cudaDevAttrMultiProcessorCount, 0);
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0);

    int mb = (m+mp-1) / mp; // ceil(m/mp)
    mb = (mb+31)/32 * 32; // mb should be divisible by 32.
    printf("Support Cooperative: %d, #SM: %d, mb: %d\n", supportsCoopLaunch, mp, mb);
    printf("m=%d, n=%d\n", m, n);
    int num_threadblocks;
    num_threadblocks = (m+mb-1) / mb;
    assert(num_threadblocks<=mp);

    void *params[8];
    params[0] = (void *) &m;
    params[1] = (void *) &n;
    params[2] = (void *) &A;
    params[3] = (void *) &lda;
    params[4] = (void *) &R;
    params[5] = (void *) &ldr;
    params[6] = (void *) &mb;
    params[7] = (void *) &work;

    assert(mb % 32 == 0);
    
    dim3 gridDim(num_threadblocks, 1, 1);
    dim3 blockDim(32, 16, 1);
    auto status = cudaLaunchCooperativeKernel((void *) mgs_panel_general_kernel, gridDim, blockDim, params);
    printf("status=%d\n", status);
    CHECK_KERNEL();
}

