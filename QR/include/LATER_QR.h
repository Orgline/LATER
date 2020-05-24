#include<LATER.h>

struct cudaCtxt {
	cublasHandle_t cublas_handle;
	cusolverDnHandle_t cusolver_handle;
};

void mgs_caqr_panel_256x128(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work);

void mgs_caqr_panel_256x32(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work);

__global__ void mgs_kernel(int m, int n, float *AA, int lda, float *RR, int ldr);