cmake_minimum_required(VERSION 3.22)
project(CppTest CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(CppTest cpu_spmma.cu)

set_target_properties(CppTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(CppTest PROPERTIES CUDA_ARCHITECTURES "80;80")

target_include_directories(CppTest
        PUBLIC ${CUDA_TOOLKIT_PATH}/include
        )

target_link_libraries(CppTest
        PUBLIC cusparse
        PUBLIC cusparseLt
        )