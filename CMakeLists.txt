cmake_minimum_required(VERSION 3.16)
project(later LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)


message("\tCUDA_PATH: $ENV{CUDA_PATH}")
message("\tCUTLASS_DIR: ${CUTLASS_DIR}")

find_library(
        CUBLAS_LIBRARY cublas
        HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        $ENV{CUBLAS_PATH}/lib64
        $ENV{CUBLAS_PATH}/lib/x64
        $ENV{CUDA_PATH}/lib64
        $ENV{CUDA_PATH}/lib/x64
        ${CUBLAS_PATH}/lib64
        ${CUBLAS_PATH}/lib/x64
        /usr/lib/x86_64-linux-gnu)
find_library(
        CUSOLVER_LIBRARY cusolver
        HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        $ENV{CUBLAS_PATH}/lib64
        $ENV{CUBLAS_PATH}/lib/x64
        $ENV{CUDA_PATH}/lib64
        $ENV{CUDA_PATH}/lib/x64
        ${CUBLAS_PATH}/lib64
        ${CUBLAS_PATH}/lib/x64
        /usr/lib/x86_64-linux-gnu)
find_library(
        CUDART_LIBRARY cudart
        HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        $ENV{CUBLAS_PATH}/lib64
        $ENV{CUBLAS_PATH}/lib/x64
        $ENV{CUDA_PATH}/lib64
        $ENV{CUDA_PATH}/lib/x64
        ${CUBLAS_PATH}/lib64
        ${CUBLAS_PATH}/lib/x64
        /usr/lib/x86_64-linux-gnu)
find_library(
        CURAND_LIBRARY curand
        HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        $ENV{CUBLAS_PATH}/lib64
        $ENV{CUBLAS_PATH}/lib/x64
        $ENV{CUDA_PATH}/lib64
        $ENV{CUDA_PATH}/lib/x64
        ${CUBLAS_PATH}/lib64
        ${CUBLAS_PATH}/lib/x64
        /usr/lib/x86_64-linux-gnu)

add_subdirectory(./QR)
add_subdirectory(util)

message("CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
include_directories(
        ./include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CUTLASS_DIR}/include
        ${CUTLASS_DIR}/tools/util/include
)

#set_source_files_properties(test.cpp PROPERTIES LANGUAGE CUDA)
#add_executable(test_cuda test.cu gpu_timer.cpp)
#set_property(TARGET test_cuda
#                 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(test_cuda ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY})
