cmake_minimum_required(VERSION 3.22)
project(Resnet CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(Resnet ./net/MyResNet.cu ./net/MyResNet.cuh
        net/entity/MyTensor.cu net/entity/MyTensor.cuh
        interface/interfaces.cu interface/interfaces.cuh resnet_main.cu
        interface/resnet_kernel.cuh interface/resnet_kernel.cu
        ../nlp/transformer/interface/kernels_transformer.cuh ../nlp/transformer/interface/kernels_transformer.cu
        ../nlp/transformer/interface/cublas_interface.cu ../nlp/transformer/interface/cublas_interface.cu
        ../nlp/transformer/interface/spmma_interface.cuh ../nlp/transformer/interface/spmma_interface.cu
        ../nlp/transformer/interface/cusparse_interface.cu ../nlp/transformer/interface/cusparse_interface.cu
        ../spmma/utils/CudaTime.cu ../spmma/utils/CudaTime.cuh test/tests.cu test/tests.cuh)

set_target_properties(Resnet PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Resnet PROPERTIES CUDA_ARCHITECTURES "80;80")

target_include_directories(Resnet
        PUBLIC ${CUDA_TOOLKIT_PATH}/include
        )

target_link_libraries(Resnet
        PUBLIC cusparseLt
        PUBLIC cublas
        PUBLIC cusparse
        PUBLIC cudnn
        )