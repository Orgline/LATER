#include "LATER.h"

/*
This function performs recursive Gram-Schmidt QR factorization

[A1|A2]=[Q1|Q2][R11|R12]
               [  0|R22]
A1=Q1*R11;
R12=Q1^T*A2;
A2=A2-Q1*R12;
A2=Q2*R22;

The input A stores the original matrix A to be factorized
the output A stores the orthogonal matrix Q
the output A stores the upper triangular matrix R

Both A and R need to be stored on GPU initially
*/

void later_rgsqrf(int m, int n, float *A, int lda, float *R, int ldr)
{
    
    return;
}