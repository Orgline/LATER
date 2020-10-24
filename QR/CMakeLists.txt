include_directories(
        ./include
)
add_library(qr
        later_rhouqr.cu
        later_rgsqrf.cu
        later_bhouqr.cu
        later_ormqr.cu
        panel.cu
        mgs_panel_general.cu
        later_oc_qr.cu)

target_include_directories(qr PUBLIC)

target_link_libraries(qr PRIVATE  blas cublas cusolver)

set_property(TARGET qr
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)

