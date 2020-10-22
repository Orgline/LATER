#include "OC_gemm.h"
#include "LATER.h"
#include <random>
#include <assert.h>

template <typename T> void prt(T *arr, int size) {
    for (int i = 0; i < size; i++)
        std::cout << arr[i] << ", ";
    puts("");
}
int main(int ac, char **av) {
    if (ac < 4) puts("Usage: ./a.out m n k [allowed GPU mem (MiB)]");
    int m = atoi(av[1]);
    int n = atoi(av[2]);
    int k = atoi(av[3]);

    size_t mem, free, total;
    cudaMemGetInfo(&free, &total);
    mem = free;
    if (ac > 4) {
        mem = atol(av[4]) * 1024 * 1024;
    }
    std::cout << mem << "\t" << free << "\t" << total << std::endl;
    assert(mem <= free);

    const size_t elements = size_t(m) * size_t(k) + size_t(n) * size_t(k);
    std::vector<half> h_data(elements);
    std::vector<float> f_data(elements);
    std::uniform_real_distribution<float> distribution(0.0f, 2.0f);
    std::mt19937 engine;
    const auto f_data_size = f_data.size();
    // #pragma omp parallel for
    //     for (size_t i = 0; i < f_data_size; i++) {
    //         f_data[i] = distribution(engine);
    //         h_data[i] = __float2half(f_data[i]);
    //     }
    auto hA = h_data.data();
    auto hB = &h_data.data()[size_t(m) * size_t(k)];
    auto fA = f_data.data();
    auto fB = &f_data.data()[size_t(m) * size_t(k)];
    std::vector<float> C_data(size_t(m) * size_t(n), 0.0f);
    auto C = C_data.data();

    float alpha = 1.0f;
    float beta = 1.0f;

    if (size_t(m) * size_t(n) < 33) {
        prt(fA, m * k);
        prt(fB, n * k);
    }

    auto pool = std::make_shared<Mem_pool>(mem); // Create memory pool
    /*
     * Usage:
     *  allocate memory: float* p = reinterpret_cast<float *>(pool->allocate(size));
     *  free memory: pool->free(p);
     */
    auto OC = new OC_gemm(m, n, k, pool);
    auto start = clock();
    OC->gemm(CUBLAS_OP_N, CUBLAS_OP_T, alpha, hA, m, hB, k, beta, C, m);
    std::cout << "Time: " << ((clock() - start) / (CLOCKS_PER_SEC / 1000)) << std::endl;
    if (size_t(m) * size_t(n) > 33)
        std::cout << C[0] << std::endl;
    else
        prt(C, m * n);
    start = clock();
    OC->gemm(CUBLAS_OP_T, CUBLAS_OP_N, alpha, fA, m, fB, k, beta, C, m);
    std::cout << "Time: " << ((clock() - start) / (CLOCKS_PER_SEC / 1000)) << std::endl;
    if (size_t(m) * size_t(n) > 33)
        std::cout << C[0] << std::endl;
    else
        prt(C, m * n);
    std::cout << pool->size() << "\n";
    delete OC;
    std::cout << pool->size() << "\n";
}