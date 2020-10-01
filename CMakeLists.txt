cmake_minimum_required(VERSION 3.17)
project(later LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
message("\tCUDA_PATH: $ENV{CUDA_PATH}")
message("\tCUTLASS_DIR: $ENV{CUTLASS_DIR}")
message("\tGPU Architecture: ${CUDA_ARCH}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -DCUDA_ARCH=${CUDA_ARCH}")

find_package(MAGMA)
if( MAGMA_FOUND ) 
    add_definitions(-DMAGMA)
endif( MAGMA_FOUND )

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    message("debug mode; enabling dbgprint")
    add_definitions(-DDBGPRINT)
    add_definitions(-DDEBUG_CUDA_KERNEL_LAUNCH)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -rdynamic -lineinfo")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)


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

add_subdirectory(QR)
add_subdirectory(BLAS)
add_subdirectory(util)
add_subdirectory(test)
add_subdirectory(benchmark)
add_subdirectory(Cholesky)

message("CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message("CUTLASS_DIR $ENV{CUTLASS_DIR}")

include_directories(
        ./include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        $ENV{CUTLASS_DIR}/include
        $ENV{CUTLASS_DIR}/tools/util/include
)

#set_source_files_properties(test.cpp PROPERTIES LANGUAGE CUDA)
#add_executable(test_cuda test.cu gpu_timer.cpp)
#set_property(TARGET test_cuda
#                 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(test_cuda ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY})
