#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>


/*
These three functions are related with QR factorization

rgsqrf: recursive Gram-Schmidt QR factorization

rhouqr: recursive Householder QR factorization

bhouqr: block Householder QR factorization
*/
void later_rgsqrf(int m, int n, float* A, int lda, float* R, int ldr, float* work, int lwork, __half* hwork, int lhwork);

void later_rhouqr(int m, int n, float* A, int lda, float* R, int ldr);

void later_bhouqr(int m, int n, float* A, int lda, float* R, int ldr);


/*
Below functions are the integration of often-used functions
*/

/*
Call startTimer() at first and then call stopTimer() to get the time consumption

Cannot be called nesting
Like:
startTimer();
    ...
    startTimer();
    ...
    stopTimer();
    ...
stopTimer();
This is not supported
*/
void startTimer();

float stopTimer();

/*
s2h: convert single matrix to half matrix
h2s: convert half matrix to single matrix
*/

__global__
void s2h(int m, int n, float *as, int ldas, __half *ah, int ldah);

__global__
void h2s(int m, int n,__half *ah, int ldah, float *as, int ldas);

/*
Generate a matrix on GPU
normal matrix has a mean 0 and a standard deviation 1
*/

void generateUniformMatrix(float *dA,int m,int n);

void generateNormalMatrix(float *dA,int m,int n);

/*
snorm returns the f-norm of a vector/matrix
*/

float snorm(int m,int n,float* dA);

/*
set a matrix to be an identity matrix
*/

__global__
void setEye( int m, int n, float *a, int lda);
