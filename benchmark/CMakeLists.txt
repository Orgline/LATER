add_executable(bench_evd bench_evd.cu)
set_property(TARGET bench_evd
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)


find_package(MAGMA REQUIRED)
include_directories(${MAGMA_INCLUDE_DIRS})


target_link_libraries(bench_evd qr utility ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY} ${CURAND_LIBRARY} ${MAGMA_LIBRARIES})

