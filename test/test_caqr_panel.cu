#include "LATER.h"
#include "LATER_QR.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <cstdio>

#ifdef WIN32
#define pclose _pclose
#define popen _popen
#endif

std::string exec(const char *cmd)
{
    std::array<char,128> buffer{};
    std::string result;
    std::unique_ptr<FILE,decltype(&pclose)> pipe(popen(cmd, "r"), pclose);

    if(!pipe) {
        printf("popen failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}
void check_qr()
{
    std::cout << "Validating QR with Julia" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << exec("julia check_qr.jl");
    std::cout << "=========================" << std::endl;
}

int main(int argc, char* argv[])
{
    float *A;
    int m=4096; // 32*256
    int n=32;
    cudaMalloc(&A, sizeof(float)*m*n);
    generateUniformMatrix(A, m, n);
    int nb = (m+255)/256;
    int r = m%256;
    int ldwork = m/256*32+32;
    int mm = m/256*32+32;
    int lda=m;

    print_env();

    printMatrixDeviceBlock<float>("A.csv",m,n,A,lda);

    float *R;
    int ldr = n;
    cudaMalloc(&R, sizeof(float)*n*n*nb);
    cudaCtxt ctxt{};
    cublasCreate(&ctxt.cublas_handle );
    cusolverDnCreate(&ctxt.cusolver_handle );
    {
        float *work;
        cudaMalloc(&work, 2*sizeof(float)*m*n);
        startTimer();
        hou_caqr_panel<256,32>(ctxt, m, n, A, lda, R, ldr, work);
        float ms = stopTimer();
        CHECK_KERNEL();
        printf("%dx%d hou_caqr_panel_256x32 block takes %.3f (ms)\n", m, n, ms);
        cudaFree(work);
    }
    printMatrixDeviceBlock("Q.csv", m, n, A, lda);
    printMatrixDeviceBlock("R.csv", n, n, R, ldr);

    check_qr();


    {
        float *work;
        cudaMalloc(&work, 2*sizeof(float)*m*n);
        generateUniformMatrix(A, m, n);

        startTimer();
        mgs_caqr_panel_256x32(ctxt, m, n, A, lda, R, ldr, work);
        float ms = stopTimer();
        CHECK_KERNEL();
        printf("%dx%d mgs_caqr_panel_256x32 block takes %.3f (ms)\n", m, n, ms);
        cudaFree(work);
    }

}