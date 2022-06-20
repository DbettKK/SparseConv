cmake_minimum_required(VERSION 3.22)
project(SparseMMA CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(SparseMMA wmma_sp.cu wmma.sp.cuh ../spmma/utils/CudaTime.cu ../spmma/utils/CudaTime.cuh cusparselt_example.cu utils/DataGenerator.cu utils/DataGenerator.cuh)

set_target_properties(SparseMMA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(SparseMMA PROPERTIES CUDA_ARCHITECTURES "80;80")

target_include_directories(SparseMMA
        PUBLIC ${CUDA_TOOLKIT_PATH}/include
        )

target_link_libraries(SparseMMA
        PUBLIC cusparse
        PUBLIC cusparseLt
        )