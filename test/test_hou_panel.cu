#define DEBUG_CUDA_KERNEL_LAUNCH

#include "LATER.h"
#include "LATER_QR.h"

int main(int argc, char* argv[])
{
    float *A;
    int m=256;
    int n=32;
    cudaMalloc(&A, sizeof(float)*m*n);
    generateUniformMatrix(A, m, n);
    int nb = (m+255)/256;
    int r = m%256;
    int ldwork = m/256*32+32;
    int mm = m/256*32+32;
    int lda=m;

    printMatrixDeviceBlock<float>("A.csv",m,n,A,lda);

    float *R;
    int ldr = n*nb;
    cudaMalloc(&R, sizeof(float)*n*n*nb);

    {
        startTimer();
        auto blockdim = dim3(32, 16);
        int nb = (m+255)/256;

        hou_kernel3<256,32><<<nb, blockdim>>>(m, n, A, lda, R, ldr);
        float ms = stopTimer();
        CHECK_KERNEL();
        printf("%dx%d hou_kernel block takes %.3f (ms)\n", m, n, ms);
    }
    printMatrixDeviceBlock("Q.csv", m, n, A, lda);
    printMatrixDeviceBlock("R.csv", n*nb, n, R, ldr);
    generateUniformMatrix(A, m, n);

    {
        startTimer();
        int nb = (m+255)/256;
        dim3 blockdim = dim3(32,32);
        mgs_kernel2<<<nb, blockdim>>>(m, n, A, lda, R, n);
        //mgs_kernel2<<<1, blockdim>>>(m, n, A,  lda, R, n);
        float ms = stopTimer();
        CHECK_KERNEL();
        printf("%dx%d mgs_kernel block takes %.3f (ms)\n", m, n, ms);
    }



}