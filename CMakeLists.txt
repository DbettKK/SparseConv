cmake_minimum_required(VERSION 3.22)
project(SparseConv CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(spmma)
add_subdirectory(ptx)
add_subdirectory(cpp)

set_target_properties(SparseConv PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
