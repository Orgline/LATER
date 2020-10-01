//
// Created by pwu on 9/30/20.
//
//#define KERNEL_LAUNCH_INFO

#include <iostream>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "LATER.h"

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

int main()
{
    cutlass::Status status;
    int M = 4096;
    int N = 4096;
    int K = 4096;

//    float alpha = 1.23f;
//    float beta = -.123f;
    print_env();
    // get CC major.minor
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);
    {

        std::cout << "*** A16B16C16C16 ***" <<
                  " " << M << "x" << N << "x" << K << std::endl;
        using ElementInputA = cutlass::half_t;
        using ElementInputB = cutlass::half_t;
        using ElementOutput = cutlass::half_t;
        cutlass::HostTensor<ElementInputA , cutlass::layout::ColumnMajor> A({M, K});
        cutlass::HostTensor<ElementInputB , cutlass::layout::ColumnMajor> B({K, N});
        cutlass::HostTensor<ElementOutput , cutlass::layout::ColumnMajor> C({M, N});
        cutlass::reference::host::TensorFillRandomUniform(
                A.host_view(),
                1,
                ElementInputA(4),
                ElementInputA(-4),
                0);  // <- Fill matrix A on host with uniform-distribution random data
        cutlass::reference::host::TensorFillRandomUniform(
                B.host_view(),
                1,
                ElementInputB(4),
                ElementInputB(-4),
                0);  // <- Fill matrix B on host with uniform-distribution random data
        cutlass::reference::host::TensorFillRandomUniform(
                C.host_view(),
                1,
                ElementOutput(4),
                ElementOutput(-4),
                0);  // <- Fill matrix C on host with uniform-distribution random data


        A.sync_device();
        B.sync_device();
        C.sync_device();
        using ElementAccumulator = cutlass::half_t;
        using ElementComputeEpilogue = ElementAccumulator;
//        using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
//                ElementOutput,
//                128 / cutlass::sizeof_bits<ElementOutput>::value,
//                ElementAccumulator,
//                ElementComputeEpilogue
//        >;
#ifdef CUDA_ARCH
#if CUDA_ARCH==Turing
#pragma message ( "Turing" )
            using SmArch = cutlass::arch::Sm75;

#elif CUDA_ARCH==Volta
 #pragma message ( "Volta" )
            using SmArch = cutlass::arch::Sm70;
#endif
#else
#error "Macro CUDA_ARCH undefined!"
#endif
        using Gemm = cutlass::gemm::device::Gemm<
                ElementInputA,
                cutlass::layout::ColumnMajor,
                ElementInputB,
                cutlass::layout::ColumnMajor,
                ElementOutput,
                cutlass::layout::ColumnMajor,
                ElementAccumulator,
                cutlass::arch::OpClassTensorOp,
                SmArch
//                cutlass::gemm::GemmShape<256,128,32>,
//                cutlass::gemm::GemmShape<64,64,32>,
//                cutlass::gemm::GemmShape<16,8,8>
//                ShapeMMAOp
//                EpilogueOutputOp
//                cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle,
//                2
        >;
        ElementAccumulator alpha = ElementAccumulator(1), beta = ElementAccumulator(-1);
//        EpilogueOutputOp::Params params(alpha, beta);
        cutlass::gemm::GemmCoord problem_size({M, N, K});
        typename Gemm::Arguments arguments{problem_size,
                                           A.device_ref(),
                                           B.device_ref(),
                                           C.device_ref(),
                                           C.device_ref(),
                                           {alpha, beta},
                                           1};
        size_t workspace_size = Gemm::get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        //auto timer = cutlass::profiler::GpuTimer();


        Gemm gemm_op;
        status = gemm_op.initialize(arguments, workspace.get());
        CUTLASS_CHECK(status);
        startTimer();
        gemm_op();
        auto ms = stopTimer();
        std::cout << "\tCUTLASS MMA takes " << ms << " (ms)";
        std::cout << "\t\t GFLOPS: " << 2.0 * M * N * K / (1.e6 * ms) << std::endl;

        startTimer();
        auto status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                                   &alpha, A.device_ref().data(), CUDA_R_16F, A.device_ref().stride(0),
                                   B.device_ref().data(), CUDA_R_16F, B.device_ref().stride(0),
                                   &beta, C.device_ref().data(), CUDA_R_16F, C.device_ref().stride(0),
                                   CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        ms = stopTimer();

        std::cout << "\tCUBLAS takes " << ms << " (ms)";
        std::cout << "\t\t GFLOPS: " << 2.0 * M * N * K / (1.e6 * ms) << std::endl;
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout << " !Unsuccessful! CUBLAS Status: " << std::endl;
            return -1;
        }
    }
#if 0
    {
        std::cout << "*** A16B16C16C32 ***" <<
                  " " << M << "x" << N << "x" << K << std::endl;

        using ElementOutput = cutlass::half_t;
        using ElementAccumulator = float;

        using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
                ElementOutput,
                128 / cutlass::sizeof_bits<ElementOutput>::value,
                ElementAccumulator,
                ElementAccumulator
        >;
        using Gemm = cutlass::gemm::device::Gemm<
                cutlass::half_t,
                cutlass::layout::ColumnMajor,
                cutlass::half_t,
                cutlass::layout::ColumnMajor,
                ElementOutput,
                cutlass::layout::ColumnMajor,
                ElementAccumulator,
                cutlass::arch::OpClassTensorOp,
                cutlass::arch::Sm70,
                cutlass::gemm::GemmShape<128,256,32>,
                cutlass::gemm::GemmShape<64,64,32>,
                cutlass::gemm::GemmShape<8,8,4>,
                EpilogueOutputOp,
                cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle,
                2
        >;
        using Gemm_wmma = cutlass::gemm::device::Gemm<
                cutlass::half_t,
                cutlass::layout::ColumnMajor,
                cutlass::half_t,
                cutlass::layout::ColumnMajor,
                ElementOutput,
                cutlass::layout::ColumnMajor,
                ElementAccumulator,
                cutlass::arch::OpClassWmmaTensorOp,
                cutlass::arch::Sm70,
                cutlass::gemm::GemmShape<128,128,32>,
                cutlass::gemm::GemmShape<64,64,32>,
                cutlass::gemm::GemmShape<16,16,16>,
                EpilogueOutputOp,
                cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle,
                2
        >;
        auto alpha = ElementAccumulator (1.23);
        auto beta = ElementAccumulator (-0.123);
        EpilogueOutputOp::Params params(alpha, beta);
        cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A({M, K});
        cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B({K, N});
        cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C({M, N});

        Gemm gemm_op;
        Gemm_wmma gemm_op_wmma;
        auto timer = cutlass::profiler::GpuTimer();
        timer.start();
        status = gemm_op({
                                 {M, N, K},
                                 A.device_ref(),
                                 B.device_ref(),
                                 C.device_ref(),
                                 C.device_ref(),
                                 {alpha, beta}
                         });
        timer.stop_and_wait();

        std::cout << "\tCUTLASS MMA takes " << timer.duration() << " (ms)";
        std::cout << "\t\t GFLOPS: " << 2.0 * M * N * K / (1.e6 * timer.duration()) << std::endl;
        if (status != cutlass::Status::kSuccess) {
            std::cout << " !Unsuccessful! CUTLASS Status: " << std::endl;
            return -1;
        }
        timer.start();
        status = gemm_op_wmma({
                                      {M, N, K},
                                      A.device_ref(),
                                      B.device_ref(),
                                      C.device_ref(),
                                      C.device_ref(),
                                      {alpha, beta}
                              });
        timer.stop_and_wait();

        std::cout << "\tCUTLASS WMMA takes " << timer.duration() << " (ms)";
        std::cout << "\t\t GFLOPS: " << 2.0 * M * N * K / (1.e6 * timer.duration()) << std::endl;
        if (status != cutlass::Status::kSuccess) {
            std::cout << " !Unsuccessful! CUTLASS Status: " << std::endl;
            return -1;
        }

        timer.start();
        auto status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                                   &alpha, A.device_ref().data(), CUDA_R_16F, A.device_ref().stride(0),
                                   B.device_ref().data(), CUDA_R_16F, B.device_ref().stride(0),
                                   &beta, C.device_ref().data(), CUDA_R_16F, C.device_ref().stride(0),
                                   CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        timer.stop_and_wait();

        std::cout << "\tCUBLAS takes " << timer.duration() << " (ms)";
        std::cout << "\t\t GFLOPS: " << 2.0 * M * N * K / (1.e6 * timer.duration()) << std::endl;
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout << " !Unsuccessful! CUBLAS Status: " << std::endl;
            return -1;
        }
    }
#endif
#if 0
    {
        std::cout << "*** A32B32C32C32 ***" << " " << M << "x" << N << "x" << K << std::endl;
        using Gemm_A32B32C32C32 = cutlass::gemm::device::Gemm<
                float,
                cutlass::layout::ColumnMajor,
                float,
                cutlass::layout::ColumnMajor,
                float,
                cutlass::layout::ColumnMajor,
                float>;
        Gemm_A32B32C32C32 gemm_op;
        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> A({M, K});
        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> B({K, N});
        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> C({M, N});

        auto timer = cutlass::profiler::GpuTimer();
        timer.start();
        status = gemm_op({
                                 {M, N, K},
                                 A.device_ref(),
                                 B.device_ref(),
                                 C.device_ref(),
                                 C.device_ref(),
                                 {alpha, beta}
                         });
        timer.stop_and_wait();
        std::cout << "\tGEMM_OP takes " << timer.duration() << " (ms)";
        std::cout << "\t\t GFLOPS: " << 2.0 * M * N * K / (1.e6 * timer.duration()) << std::endl;
        if (status != cutlass::Status::kSuccess) {
            std::cout << " !Unsuccessful! CUTLASS Status: " << std::endl;
            return -1;
        }

        timer.start();
        auto status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                                   &alpha, A.device_ref().data(), CUDA_R_32F, A.device_ref().stride(0),
                                   B.device_ref().data(), CUDA_R_32F, B.device_ref().stride(0),
                                   &beta, C.device_ref().data(), CUDA_R_32F, C.device_ref().stride(0),
                                   CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        timer.stop_and_wait();

        std::cout << "\tCUBLAS takes " << timer.duration() << " (ms)";
        std::cout << "\t\t GFLOPS: " << 2.0 * M * N * K / (1.e6 * timer.duration()) << std::endl;
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout << " !Unsuccessful! CUBLAS Status: " << std::endl;
            return -1;
        }
    }
    std::cout << "\n";
    std::cout << "=== Test CUTLASS functionality: THREADBLOCK ===" << std::endl;



    std::cout << "=== Test cuSOLVER QR ===" << std::endl;

    {
        int M=10000, K = 10000;
        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> A({M, K});
        assert(A.device_data() == A.device_ref().data());
        int lwork;
        auto cusolver_status = cusolverDnSgeqrf_bufferSize(
                cusolver_handle,
                M, K, A.device_data(), A.device_ref().stride(0),&lwork);

        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> tau({M, 1});
        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> workspace({lwork,1});
        //cutlass::HostTensor<int, cutlass::layout::ColumnMajor> devInfo({1,1});
        int *devInfo;
        cudaMalloc(&devInfo, sizeof(int));
        //std::cout << "DEBUG A.device

        auto timer = cutlass::profiler::GpuTimer();
        timer.start();
        cusolver_status = cusolverDnSgeqrf(cusolver_handle, M, K, A.device_data(), A.device_ref().stride(0),
                tau.device_data(), workspace.device_data(), lwork, devInfo);
        timer.stop_and_wait();
        std::cout <<  M << "x" << K << "\tSGEQRF takes " << timer.duration() << " (ms)";
        std::cout << "\t\t GFLOPS: " << 2*(1.0*M*M*M-1.0/3.0*M*K*K) / (1.e6 * timer.duration()) << std::endl;
        std::cout << "DEBUG cusolver_status=" << cusolver_status << std::endl;
        int info;
        cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "DEBUG devInfo=" << info << std::endl;
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    }

    std::cout << "=== Test cuSOLVER LU ===" << std::endl;

    {
        int M = 10000;
        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> A({M, M});
        cutlass::reference::device::TensorFillRandomGaussian( A.device_view(), 0x2019, 0.0f, 1.0f);
        int lwork;
        auto cusolver_status = cusolverDnSgetrf_bufferSize(
                cusolver_handle,
                M, M, A.device_data(), A.device_ref().stride(0),&lwork);
        std::cout << "DEBUG: lwork=" << lwork << std::endl;
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

        cutlass::HostTensor<int, cutlass::layout::ColumnMajor> ipiv({M, 1});
        cutlass::HostTensor<float, cutlass::layout::ColumnMajor> workspace({lwork,1});
        cutlass::HostTensor<int, cutlass::layout::ColumnMajor> devInfo({1,1});

        auto timer = cutlass::profiler::GpuTimer();
        timer.start();
        cusolver_status = cusolverDnSgetrf(cusolver_handle, M, M, A.device_data(), A.device_ref().stride(0),
                workspace.device_data(), ipiv.device_data(),  devInfo.device_data());
        timer.stop_and_wait();
        std::cout << M << "x" << M << "\tSGETRF takes " << timer.duration() << " (ms)";
        std::cout << "\t\t GFLOPS: " << 2.0/3.0*M*M*M / (1.e6 * timer.duration()) << std::endl;
        std::cout << "DEBUG cusolver_status=" << cusolver_status << std::endl;
        devInfo.sync_host();
        std::cout << "DEBUG devInfo=" << devInfo.host_ref().at({0,0}) << std::endl;
        assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
        ipiv.sync_host();
        std::cout << "IPIV 1 2 3=" <<  ipiv.at({0,0}) << " " << ipiv.at({1,0}) << std::endl;
    }
#endif
    return 0;


}