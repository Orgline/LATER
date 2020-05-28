#pragma once

#include<LATER.h>

struct cudaCtxt {
	cublasHandle_t cublas_handle;
	cusolverDnHandle_t cusolver_handle;
};

struct F4add
{
    __host__ __device__ __forceinline__
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
};



void mgs_caqr_panel_256x128(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work);

void mgs_caqr_panel_256x32(cudaCtxt ctxt, int m, int n, float *A, int lda, float *R, int ldr, float *work);

__global__ void mgs_kernel(int m, int n, float *AA, int lda, float *RR, int ldr);
__global__ void mgs_kernel2(int m, int n, float *AA, int lda, float *RR, int ldr);