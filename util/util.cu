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

// the following are for getting system (GPU, CUDA, OS, Compiler) information
std::string getOsName()
{
#ifdef _WIN64
    return "Windows 64-bit";
#elif _WIN32
    return "Windows 32-bit";
#elif __APPLE__ || __MACH__
    return "Mac OSX";
#elif __linux__
    return "Linux";
#elif __FreeBSD__
    return "FreeBSD";
    #elif __unix || __unix__
    return "Unix";
    #else
    return "Other";
#endif
}
std::string getCompilerName()
{
#ifdef _MSC_VER
    return "Visual Studio " + std::to_string(_MSC_VER);
#elif __GNUC__
    std::stringstream ss;
    ss << "GCC " <<  __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
    return ss.str();
#elif __clang__
    return "Clang";
#endif
    return "Unkonwn";
}
void print_env() {
    cudaDeviceProp prop;
    int cudaversion;
    int driverversion;

    cudaGetDeviceProperties(&prop, 0);
    int mpcount, s2dratio;
    cudaDeviceGetAttribute(&mpcount, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&s2dratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, 0);
    cudaDeviceGetAttribute(&mpcount, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&mpcount, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&mpcount, cudaDevAttrMultiProcessorCount, 0);
    cudaRuntimeGetVersion(&cudaversion);
    cudaDriverGetVersion(&driverversion);

    std::cout << "=== Device information ===" << std::endl;
    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "OS: " << getOsName() << std::endl;
    std::cout << "Host Compiler: " << getCompilerName() << std::endl;
    std::cout << "CUDA Runtime Version: " << cudaversion << std::endl;
    std::cout << "CUDA Driver Version: " << driverversion << std::endl;
    std::cout << "NVCC Version: " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << std::endl;
    std::cout << "GMem " << prop.totalGlobalMem << std::endl;
    std::cout << "SMem per block " << prop.sharedMemPerBlock << std::endl;
    std::cout << "SMem per MP " << prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Regs per block " << prop.regsPerBlock << std::endl;
    std::cout << "Clock rate " << prop.clockRate << std::endl;
    std::cout << "L2 $ size " << prop.l2CacheSize << std::endl;
    std::cout << "# MP " << mpcount << std::endl;
    std::cout << "single-double perf ratio " << s2dratio << std::endl;
//    std::cout << "__CUAD_ARCH__ " << __CUDA_ARCH__ << std::endl;
    std::cout << "=== END Deivce Information ===\n" << std::endl;
}