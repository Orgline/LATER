#include "LATER.h"
#include "LATER_QR.h"

/*
This routine performs block Householder QR factorization. It uses rhouqr as panel factorization.
The input A stores the original matrix A that will be factorized
The output A stores the Householder vectors Y
The output W stores the W matrix of WY representation
THe output R stores the upper triangular matrix
*/

#define NMIN 512

bool wflag = false;


void printMatrixDeviceBlock_(char *filename,int m, int n, float *dA, int lda)
{
    FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("fault!\n");
		return;
	}
    //printf("Perform printmatrixdevice\n");
    float *ha;
    ha = (float*)malloc(sizeof(float));

    for(int i = 0;i<m;i++)
    {
        for(int j = 0;j<n;j++)
        {
            cudaMemcpy(&ha[0], &dA[i+j*lda], sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(f, "%lf", ha[0]);
            if (j == n - 1) fprintf(f, "\n");
			else fprintf(f, ",");
        }
    }
    fclose(f);
	//cudaMemcpy(ha, dA, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    //printMatrixFloat(filename, m, n, ha, lda);
    free(ha);
}

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

void formW(int m, int n, float* W, int ldw, float* Y, int ldy, float *work)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    float sone = 1.0;
    float snegone = -1.0;
    float szero = 0.0;

    cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n/2,n/2,m,
        &sone,
        Y, ldy,
        W+ldw/2*n,ldw,
        &szero,
        work,n/2
    );

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m,n/2,n/2,
        &snegone,
        W, ldw,
        work,n/2,
        &sone,
        W+ldw/2*n,ldw
    );
}

float panelTime = 0.0;
float gemmTime = 0.0;

void later_bhouqr(int m, int n, float* A, int lda, float* W, int ldw, float* R, int ldr, float* work, int lwork, __half* hwork, int lhwork, float* U)
{
    printf("Function bhouqr\n");
    
    cudaCtxt ctxt;
    cublasCreate( & ctxt.cublas_handle );
    cusolverDnCreate( & ctxt.cusolver_handle );

    float sone = 1.0;
    float snegone = -1.0;
    float szero = 0.0;

    //printMatrixDeviceBlock_("A.csv", m, n, A, lda);

    for(int i = 0; i < n; i += NMIN)
    {
        int nb = min(NMIN, n-i);

        //printf("n = %d\n, nb = %d\n, NMIN = %d\n", n, nb, NMIN);

        //printMatrixDeviceBlock_("A1.csv", m-i, nb, A+i*lda+i, lda);
        //printMatrixDeviceBlock_("oA2.csv", m - i,n - i - nb, A+(i+nb)*lda+i, lda);

        //panel factorization
        startTimer();
        later_rhouqr(m-i, nb, A+i*lda+i, lda, W+i*lda+i, ldw, R+i*ldr+i, ldr, work, lwork, hwork, lhwork, U);
        
        formW(m-i, nb, W+i*lda+i, ldw, A+i*lda+i, lda, work);
        panelTime += stopTimer();
        
        //printMatrixDeviceBlock_("W.csv", m-i, nb, W+i*lda+i, ldw);
        //printMatrixDeviceBlock_("Y.csv", m-i, nb, A+i*lda+i, lda);
        //printMatrixDeviceBlock_("R.csv", nb, nb, R+i*ldr+i, ldr);

        //trailing matrix update
        startTimer();
        if(n-i > NMIN)
        {
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
            //printMatrixDeviceBlock_("A2.csv", m - i,n - i - nb, A+(i+nb)*lda+i, lda);
            //break;
    
            dim3 grid( (nb+1)/32, (n-i-nb+1)/32 );
            dim3 block( 32, 32 );
            copyAndClear<<<grid, block>>>(nb, n - i - nb, A+(i+nb)*lda+i, lda, R+(i+nb)*ldr+i, ldr); 
        }
    

        //update W
        if(i!=0 && wflag)
        {
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
        gemmTime += stopTimer();
        //printMatrixDeviceBlock_("WW.csv", m, n, W, ldw);
        //printMatrixDeviceBlock_("YY.csv", m, n, A, lda);
        
    }
    //printMatrixDeviceBlock_("W.csv", m, n, W, ldw);
    //printMatrixDeviceBlock_("Y.csv", m, n, A, lda);
    //printMatrixDeviceBlock_("R.csv", n, n, R, ldr);
    printf("Panel takes %lf\n", panelTime);
    printf("Gemm takes %lf\n", gemmTime);
    cublasDestroy(ctxt.cublas_handle);
    cusolverDnDestroy(ctxt.cusolver_handle);
}
