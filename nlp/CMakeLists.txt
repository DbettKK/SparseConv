cmake_minimum_required(VERSION 3.22)
project(Nlp CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(Nlp ./interface/sparse_mma_gemm.cu ./interface/sparse_mma_gemm.cuh nlp_main.cu)

set_target_properties(Nlp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Nlp PROPERTIES CUDA_ARCHITECTURES "80;80")

target_include_directories(Nlp
        PUBLIC ${CUDA_TOOLKIT_PATH}/include
        )

target_link_libraries(Nlp
        PUBLIC cusparse
        PUBLIC cusparseLt
        )