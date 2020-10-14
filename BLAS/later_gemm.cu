#include <bits/stdc++.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "LATER.h"
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

constexpr auto stream_num = 4;
static cudaStream_t streams[stream_num];
static cublasHandle_t handles[stream_num];
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

/*
size_t free_mem() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}*/

int is_mul_overflow(int a, int b) {
    if (a >= 0 && b >= 0) {
        return INT_MAX / a < b;
    } else if (a < 0 && b < 0) {
        return INT_MAX / a > b;
    } else if (a * b == INT_MIN) {
        return 0;
    } else {
        return a < 0 ? is_mul_overflow(-a, b) : is_mul_overflow(a, -b);
    }
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
        if (is_mul_overflow(tn, sizeof(float)) || is_mul_overflow(tm, sizeof(float)) ||
            is_mul_overflow(tk, sizeof(float)) || is_mul_overflow(tm, tn * sizeof(float)) ||
            is_mul_overflow(tm, tk * sizeof(float)) || is_mul_overflow(tn, tk * sizeof(float)))
            continue;
        if (tm * tn + tm * tk + tn * tk < free_entries) break;
    } while (true);
}

// col-major
void later_oc_sgemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
              const float &alpha, const float *A, int lda, const float *B, int ldb,
              const float &beta, float *C, int ldc) {
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
                int tlda, tldb;
                const float *pA, *pB;
                if (transa == CUBLAS_OP_N) {
                    pA = &A[a * tk * m + i * tm];
                    cublasChk(cublasSetMatrixAsync(tm, tk, sizeof(float), pA, m, A_tiles[stream_id],
                                                   tm, stream));
                    tlda = tm;
                } else {
                    pA = &A[i * tk * m + a * tm];
                    cublasChk(cublasSetMatrixAsync(tk, tm, sizeof(float), pA, k, A_tiles[stream_id],
                                                   tk, stream));
                    tlda = tk;
                }
                if (transb == CUBLAS_OP_N) {
                    pB = &B[j * tn * k + a * tk];
                    cublasChk(cublasSetMatrixAsync(tk, tn, sizeof(float), pB, k, B_tiles[stream_id],
                                                   tk, stream));
                    tldb = tk;
                } else {
                    pB = &B[a * tn * k + j * tk];
                    cublasChk(cublasSetMatrixAsync(tn, tk, sizeof(float), pB, n, B_tiles[stream_id],
                                                   tn, stream));
                    tldb = tn;
                }
                cublasSgemm(handles[stream_id], transa, transb, tm, tn, tk, &alpha,
                            A_tiles[stream_id], tlda, B_tiles[stream_id], tldb, &beta,
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

/*
#ifdef TEST_OC
template <typename T> using arr_t = std::unique_ptr<T[]>;
arr_t<half> rand_half(size_t size) {
    arr_t<half> res(new half[size]);
    for (long i = 0; i < size; i++)
        res[i] = __float2half(static_cast<float>((rand() % 10) / 5.0f));
    return std::move(res);
}
arr_t<float> rand_float(size_t size) {
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
template <typename T> void prt(T *arr, int size) {
    for (int i = 0; i < size; i++)
        std::cout << arr[i] << ", ";
    puts("");
}
using namespace std::chrono;
int main(int ac, char **av) {
    cudaChk(cudaFree(0));
    if (ac < 4) puts("Usage: ./a.out m n k [used GPU mem%]");
    int m = atoi(av[1]);
    int n = atoi(av[2]);
    int k = atoi(av[3]);
    if (ac > 4) {
        size_t mem = free_mem() * (atof(av[4]) / 100);
        volatile char *p;
        cudaChk(cudaMalloc((void **)&p, mem));
    }
    const size_t elements = size_t(m) * size_t(k) + size_t(n) * size_t(k);
    std::vector<float> data(elements);
    std::uniform_real_distribution<float> distribution(0.0f, 2.0f);
    std::mt19937 engine;
    auto generator = std::bind(distribution, engine);
    std::generate_n(data.begin(), elements, generator);
    auto A = data.data();
    auto B = &data.data()[size_t(m) * size_t(k)];
    arr_t<float> C(new float[size_t(m) * size_t(n)]);
    float alpha = 1.0f, beta = 1.0f;
    OC_Sgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, m, B, k, beta, C.get(), m);
    std::cout << C[0] << std::endl;
}
#endif*/
