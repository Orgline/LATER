#include "LATER.h"

cudaEvent_t begin, end;
void startTimer()
{
    cudaEventCreate(&begin);
    cudaEventRecord(begin);
    cudaEventCreate(&end);
}

float stopTimer()
{
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds;
}

__global__
void s2h(int m, int n, float *as, int ldas, __half *ah, int ldah)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		ah[i + j*ldah] = __float2half(as[i + j*ldas]);
	}
}

__global__
void h2s(int m, int n,__half *ah, int ldah, float *as, int ldas)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		as[i + j*ldah] = __half2float(ah[i + j*ldas]);
	}
}

void generateNormalMatrix(float *dA,int m,int n)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = rand()%3000;
	curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, dA, m*n,0,1);
}

void generateUniformMatrix(float *dA,int m,int n)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = 3000;
	curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen,dA,m*n);
}

float snorm(int m,int n,float* dA)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float sn;
    int incx = 1;
    cublasSnrm2(handle, m*n, dA, incx, &sn);
    cublasDestroy(handle);
    return sn;
}

__global__
void setEye( int m, int n, float *a, int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		if (i == j) 
			a[i+j*lda] = 1;
		else
			a[i+j*lda] = 0;
	}
}

void sSubstract(cublasHandle_t handle, int m,int n, float* dA,int lda, float* dB, int ldb)
{

    float snegone = -1.0;
    float sone = 1.0;
    cublasSgeam(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n,
        &snegone,
        dA, lda,
        &sone,
        dB, ldb,
        dA, lda);
}

__global__
void deviceCopy( int m, int n, float *da, int lda, float *db, int ldb )
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		db[i+j*ldb] = da[i+j*lda];
	}
}

__global__
void clearTri(char uplo, int m, int n, float *a, int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		if (uplo == 'l') {
			if (i>j) {
				a[i+j*lda] = 0;
			}
        } 
        else
        {
            if (i<j)
                a[i+j*lda] = 0;
		}
	}
}