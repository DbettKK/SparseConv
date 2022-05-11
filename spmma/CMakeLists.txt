cmake_minimum_required(VERSION 3.22)
project(SparseConv CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(SparseConv
        entity/MatrixParam.cu entity/MatrixParam.cuh
        entity/Tensor4d.cu entity/Tensor4d.cuh
        entity/ConvParam.cu entity/ConvParam.cuh
        kernels/sparse_conv.cuh kernels/sparse_conv.cu
        kernels/sparse_matmul.cuh kernels/sparse_matmul.cu
        utils/util.cuh  utils/util.cu
        utils/CudaTime.cu utils/CudaTime.cuh test/test.cu
        kernels/kernels.cuh utils/const.cuh
        test/test_network.cu)

set_target_properties(SparseConv PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(SparseConv
        PUBLIC ${CUDA_TOOLKIT_PATH}/include
        )

target_link_libraries(SparseConv
        PUBLIC cusparse
        PUBLIC cusparseLt
        )