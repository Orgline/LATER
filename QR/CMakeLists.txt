add_library(qr later_rhouqr.cu later_rgsqrf.cu later_bhouqr.cu later_ormqr.cu panel.cu later_oc_qr.cu ./include/LATER_QR.h)

target_include_directories(qr PUBLIC ../include ./include)

target_link_libraries(qr PRIVATE  blas cublas cusolver)

set_property(TARGET qr
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)

