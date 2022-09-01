cmake_minimum_required(VERSION 3.22)
project(Test CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(Test test_main.cu ../spmma/utils/CudaTime.cuh ../spmma/utils/CudaTime.cu test_event.cu test_event.cuh)

set_target_properties(Test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Test PROPERTIES CUDA_ARCHITECTURES "80;80")

target_include_directories(Test
        PUBLIC ${CUDA_TOOLKIT_PATH}/include
        )

target_link_libraries(Test
        PUBLIC cusparseLt
        PUBLIC cublas
        PUBLIC cusparse
        PUBLIC cudnn
        )