cmake_minimum_required(VERSION 3.21)
project(Im2colTest CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(im2col_test im2col_test.cu ../entity/ConvParam.cuh ../entity/ConvParam.cu ../entity/MatrixParam.cu ../entity/MatrixParam.cuh ../entity/Tensor4d.cu ../entity/Tensor4d.cuh ../utils/util.cu ../utils/util.cuh ../utils/CudaTime.cu ../utils/CudaTime.cuh)

set_target_properties(im2col_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(im2col_test
        PUBLIC ${CUDA_TOOLKIT_PATH}/include
)

target_link_libraries(im2col_test
        PUBLIC cudnn
)

# nvcc im2col_test.cu ../entity/ConvParam.cu ../entity/MatrixParam.cu ../entity/Tensor4d.cu ../utils/util.cu ../utils/CudaTime.cu -lcudnn