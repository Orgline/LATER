add_library(chol later_rpotrf.cu)

target_include_directories(chol PUBLIC ../include)

target_link_libraries(chol PRIVATE cublas cusolver)

set_property(TARGET chol
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)
