include_directories(
        ../QR/include
)

file(GLOB TEST_SOURCES "*.cu")

foreach(SRC ${TEST_SOURCES})
    get_filename_component(NAME ${SRC} NAME_WE)
    message(${NAME})
    add_executable(${NAME} "${NAME}.cu")
    set_property(TARGET ${NAME}
            PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${NAME} qr utility ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY})
endforeach()
