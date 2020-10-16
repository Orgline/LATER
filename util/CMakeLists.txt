add_library(utility util.cu mem_pool.cu ../include/LATER.h ../include/mem_pool.h)

target_include_directories(utility PUBLIC ../include ./include)

target_link_libraries(utility PRIVATE cublas curand)


set_property(TARGET utility
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)