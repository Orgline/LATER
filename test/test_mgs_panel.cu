#include "LATER.h"
#include "LATER_QR.h"

int main(int argc, char* argv[])
{
    float *A;
    int m=256;
    int n=32;
    int lda=256;
    cudaMalloc(&A, sizeof(float)*m*n);
    generateUniformMatrix(A, m, n);
    int nb = m/256;
    int r = m%256;
    int ldwork = m/256*32+32;
    int mm = m/256*32+32;
    printMatrixDeviceBlock<float>("A.csv",m,n,A,lda);

    float *R;
    cudaMalloc(&R, sizeof(float)*n*n);

    {
        startTimer();
        auto blockdim = dim3(32, 32);
        //mgs_kernel<<<1, 256>>>(m, n, A,  lda, R, n);
        mgs_kernel2<<<1, blockdim>>>(m, n, A, lda, R, n);
        float ms = stopTimer();
        gpuErrchk( cudaPeekAtLastError() );
        printf("256*32 panel2 block takes %.3f (ms)\n", ms);
    }
    printMatrixDeviceBlock("Q.csv", m, n, A, lda);
    printMatrixDeviceBlock("R.csv", n, n, R, n);
    generateUniformMatrix(A, m, n);

    {
        startTimer();
        auto blockdim = dim3(32, 32);
        mgs_kernel<<<1, 256>>>(m, n, A, lda, R, n);
        //mgs_kernel2<<<1, blockdim>>>(m, n, A,  lda, R, n);
        float ms = stopTimer();
        printf("256*32 panel1 block takes %.3f (ms)\n", ms);
    }



}