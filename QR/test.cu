#include "LATER.h"

int main()
{
    int m,n,lda,ldr;
    float *A,*R;
    later_rhouqr(m,n,A,lda,R,ldr);
    return 0;
}