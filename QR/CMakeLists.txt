add_library(qr later_rhouqr.cu later_rgsqrf.cu later_bhouqr.cu panel.cu ./include/LATER_QR.h)

target_include_directories(qr PUBLIC ../include ./include)

target_link_libraries(qr PRIVATE cublas)

set_source_files_properties(test.cpp PROPERTIES LANGUAGE CUDA)
add_executable(test_qr test.cu)
set_property(TARGET test_qr
                 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET qr
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_qr qr utility ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY})
