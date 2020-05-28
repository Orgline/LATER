add_executable(test_qr test.cu)
set_property(TARGET test_qr
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET qr
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_qr qr utility ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY})
