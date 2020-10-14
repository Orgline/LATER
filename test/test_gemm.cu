#include "LATER.h"
#include <bits/stdc++.h>
#include <stdlib.h>

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

float ref(float *A, float *B, float *C, float *C2, int m, int n, int k) {
    float fnorm = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float temp = 0;
            for (int a = 0; a < k; a++) {
                temp += A[i + a * m] * B[a + j * k];
            }
            C[i + j * m] = temp;
            fnorm = (C[i + j * m] - C2[i + j * m])*(C[i + j * m] - C2[i + j * m]);
        }
        if(i%10 == 0)
            printf("%d\n", i);
    }
    return fnorm;
}
using namespace std::chrono;
int main(int ac, char **av) {
    if (ac < 4) puts("Usage: ./a.out m n k [used GPU mem%]");
    int m = atoi(av[1]);
    int n = atoi(av[2]);
    int k = atoi(av[3]);
    printf("m,n,k=%d,%d,%d\n", m, n, k);
    if (ac > 4) {
        size_t mem = free_mem() * (atof(av[4]) / 100);
        volatile char *p;
        printf("memory size %d\n", mem/1024/1024);
        cudaMalloc((void **)&p, mem);
    }
    arr_t<float> A = rand_float(m * k);
    arr_t<float> B = rand_float(n * k);
    arr_t<float> C(new float[m * n]), C2(new float[m * n]);
    float alpha = 1.0f, beta = 0.0f;

    later_oc_sgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A.get(), m, B.get(), k, beta, C.get(), m);

    printf("Error is %f\n",ref(A.get(), B.get(), C2.get(), C.get(), m ,n, k)/m/n);

    /*
    prt(A.get(), m * k);
    prt(B.get(), n * k);
    later_oc_sgemm(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A.get(), m, B.get(), k, beta, C.get(), m);
    prt(C.get(), m * n);
    later_oc_sgemm(CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, A.get(), m, B.get(), k, beta, C.get(), m);
    prt(C.get(), m * n);
    later_oc_sgemm(CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, A.get(), m, B.get(), k, beta, C.get(), m);
    prt(C.get(), m * n);
    later_oc_sgemm(CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, A.get(), m, B.get(), k, beta, C.get(), m);
    prt(C.get(), m * n);*/

    // double time = 0.0;
    // for (int i = 0; i < 10; i++) {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     OC_Sgemm(m, n, k, alpha, A.get(), m, B.get(), k, beta, C.get(), m);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1.0;
    // }
    // std::cout << "Time: " << time / 10 << "ms" << std::endl;
    // std::cout << C[0] << std::endl;

    // ref(A.get(), B.get(), C2.get(), m, n, k);
    // for (long i = 0; i < m * n; i++) {
    //     // std::cout << C[i] << "\t" << C2[i] << std::endl;
    //     if (abs(C[i] - C2[i]) > 1e-3)
    //         std::cout << "error at " << i << ": " << C[i] << " " << C2[i] << std::endl;
    // }
}