include_directories(
        ../QR/include
)
 add_library(evd later_qdwh_polar.cu later_sy2sb_blocked.cu)

target_include_directories(evd PUBLIC ../include)

target_link_libraries(evd PRIVATE blas cublas cusolver)

set_property(TARGET evd
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)

