add_library(blas later_rtrsm.cu later_rsyrk.cu)

target_include_directories(blas PUBLIC ../include)

target_link_libraries(blas PRIVATE cublas cusolver)

set_property(TARGET blas
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)