#include <bits/stdc++.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#define cudaChk(stat)                                                                              \
    { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
        exit(1);
    }
}
const char *cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}
#define cublasChk(stat)                                                                            \
    { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %s %s %d\n", cublasGetErrorString(stat), file, line);
        exit(1);
    }
}

using handle_t = cublasHandle_t;
using T_op_t = cublasOperation_t;

constexpr auto stream_num = 4;
static cudaStream_t streams[stream_num];
static handle_t handles[stream_num];
void init() {
    static bool first_time = true;
    if (first_time) {
        first_time = false;
        for (int i = 0; i < stream_num; i++) {
            cudaChk(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
            cublasChk(cublasCreate(&handles[i]));
            cublasChk(cublasSetStream(handles[i], streams[i]));
        }
    }
}

size_t free_mem() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

template <typename T>
void tile_size(const int m, const int n, const int k, int &tm, int &tn, int &tk) {
    auto free_entries = (free_mem() / sizeof(T)) / stream_num;
    int i = 1;
    do {
        tm = m / i;
        tn = n / i;
        tk = k / i;
        i *= 2;
    } while (tm * tk + tk * tn + tm * tn > free_entries);
}

// col-major only
void OC_Sgemm(int m, int n, int k, const float &alpha, const float *A, int lda, const float *B,
              int ldb, const float &beta, float *C, int ldc) {
    init(); // init stream and cublas handle
    int tm, tn, tk;
    tile_size<float>(m, n, k, tm, tn,
                     tk); // get the size of segment residing on GPU based on free memory
    float *A_tiles[stream_num];
    float *B_tiles[stream_num];
    float *C_tiles[stream_num];
    for (int i = 0; i < stream_num; i++) { // allocate device memory here.
        cudaChk(cudaMalloc((void **)&A_tiles[i], tm * tk * sizeof(float)));
        cudaChk(cudaMalloc((void **)&B_tiles[i], tn * tk * sizeof(float)));
        cudaChk(cudaMalloc((void **)&C_tiles[i], tn * tm * sizeof(float)));
    }
    for (int i = 0; i < (m / tm); i++) {
        for (int j = 0; j < (n / tn); j++) {
            const auto stream_id = (i * n / tn + j) % stream_num;
            auto stream = streams[stream_id];
            auto pC = &C[j * tn * m + i * tm];
            for (int a = 0; a < (k / tk); a++) {
                auto pA = &A[a * tk * m + i * tm];
                auto pB = &B[j * tn * k + a * tk];
                cublasChk(cublasSetMatrixAsync(tm, tk, sizeof(float), pA, m, A_tiles[stream_id], tm,
                                               stream));
                cublasChk(cublasSetMatrixAsync(tk, tn, sizeof(float), pB, k, B_tiles[stream_id], tk,
                                               stream));
                cublasSgemm(handles[stream_id], CUBLAS_OP_N, CUBLAS_OP_N, tm, tn, tk, &alpha,
                            A_tiles[stream_id], tm, B_tiles[stream_id], tk, &beta,
                            C_tiles[stream_id], tm);
            }
            cublasChk(
                cublasGetMatrixAsync(tm, tn, sizeof(float), C_tiles[stream_id], tm, pC, m, stream));
            cudaChk(cudaMemsetAsync(C_tiles[stream_id], 0, tm * tn * sizeof(float), stream));
        }
    }
    cudaChk(cudaDeviceSynchronize());
    for (int i = 0; i < stream_num; i++) {
        cudaChk(cudaFree(A_tiles[i]));
        cudaChk(cudaFree(B_tiles[i]));
        cudaChk(cudaFree(C_tiles[i]));
    }
}

#ifdef TEST_OC
template <typename T> using arr_t = std::unique_ptr<T[]>;
arr_t<half> rand_half(long size) {
    arr_t<half> res(new half[size]);
    for (long i = 0; i < size; i++)
        res[i] = __float2half(static_cast<float>((rand() % 10) / 5.0f));
    return std::move(res);
}

arr_t<float> rand_float(long size) {
    arr_t<float> res(new float[size]);
    for (long i = 0; i < size; i++)
        res[i] = static_cast<float>((rand() % 10) / 5.0f);
    return std::move(res);
}

void ref(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float temp = 0;
            for (int a = 0; a < k; a++) {
                temp += A[i + a * m] * B[a + j * k];
            }
            C[i + j * m] = temp;
        }
    }
}
using namespace std::chrono;
int main(int ac, char **av) {
    if (ac < 4) puts("Usage: ./a.out m n k [used GPU mem%]");
    int m = 1 << atoi(av[1]);
    int n = 1 << atoi(av[2]);
    int k = 1 << atoi(av[3]);
    if (ac > 4) {
        size_t mem = free_mem() * (atof(av[4]) / 100);
        volatile char *p;
        cudaChk(cudaMalloc((void **)&p, mem));
    }
    arr_t<float> A = rand_float(m * k);
    arr_t<float> B = rand_float(n * k);
    arr_t<float> C(new float[m * n]), C2(new float[m * n]);
    float alpha = 1.0f, beta = 1.0f;
    double time = 0.0;
    OC_Sgemm(m, n, k, alpha, A.get(), m, B.get(), k, beta, C.get(), m);
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        OC_Sgemm(m, n, k, alpha, A.get(), m, B.get(), k, beta, C.get(), m);
        auto end = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1.0;
    }
    std::cout << "Time: " << time / 10 << "ms" << std::endl;
    std::cout << C[0] << std::endl;
    // ref(A.get(), B.get(), C2.get(), m, n, k);
    // for (long i = 0; i < m * n; i++) {
    //     // std::cout << C[i] << "\t" << C2[i] << std::endl;
    //     if (abs(C[i] - C2[i]) > 1e-3)
    //         std::cout << "error at " << i << ": " << C[i] << " " << C2[i] << std::endl;
    // }
}
#endif