#include "LATER.h"
#include "LATER_QR.h"

/*
This routine performs block Householder QR factorization. It uses rhouqr as panel factorization.
The input A stores the original matrix A that will be factorized
The output A stores the Householder vectors Y
The output W stores the W matrix of WY representation
THe output R stores the upper triangular matrix
*/

#define NMIN 128




__global__
void copyAndClear( int m, int n, float *da, int lda, float *db, int ldb )
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
        db[i+j*ldb] = da[i+j*lda];
        da[i+j*lda] = 0.0;
	}
}

void later_bhouqr(int m, int n, float* A, int lda, float* W, int ldw, float* R, int ldr, float* work, int lwork, __half* hwork, int lhwork, float* U)
{
    printf("Function bhouqr\n");
    
    cudaCtxt ctxt;
    cublasCreate( & ctxt.cublas_handle );
    cusolverDnCreate( & ctxt.cusolver_handle );

    float sone = 1.0;
    float snegone = -1.0;
    float szero = 0.0;

    for(int i=0; i<n; i+=NMIN)
    {
        int nb = min(NMIN, n-i);

        //panel factorization
        later_rhouqr(m-i, nb, A+i*lda+i, lda, W+i*lda+i, ldw, R+i*ldr+i, ldr, work, lwork, hwork, lhwork, U);

        //trailing matrix update
        cublasSgemm(ctxt.cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            nb,n - i - nb, m - i,
            &sone,
            W+i*lda+i, ldw,
            A+(i+nb)*lda+i, lda,
            &szero,
            work, nb
        );

        cublasSgemm(ctxt.cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m - i,n - i - nb, nb,
            &snegone,
            A+i*lda+i, lda,
            work,nb,
            &sone,
            A+(i+nb)*lda+i,lda
        );

        dim3 grid( (nb+1)/32, (nb+1)/32 );
        dim3 block( 32, 32 );
        copyAndClear<<<grid, block>>>(nb, n - i - nb, A+(i+nb)*lda+i, lda, R+(i+nb)*ldr+i, ldr); 

        //update W
        cublasSgemm(ctxt.cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            i, nb, m,
            &sone,
            A, lda,
            W+i*lda, ldw,
            &szero,
            work, i
        );

        cublasSgemm(ctxt.cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, nb, i,
            &snegone,
            W, ldw,
            work,i,
            &sone,
            W+i*lda,ldw
        );
    }

    cublasDestroy(ctxt.cublas_handle);
    cusolverDnDestroy(ctxt.cusolver_handle);
}