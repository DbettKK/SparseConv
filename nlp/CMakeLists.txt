cmake_minimum_required(VERSION 3.22)
project(Nlp CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(Nlp nlp_main.cu
        ../spmma/utils/CudaTime.cu ../spmma/utils/CudaTime.cuh
        transformer/Encoder.cu transformer/Encoder.cuh transformer/Attention.cu transformer/Attention.cuh transformer/MatrixHalf.cu transformer/MatrixHalf.cuh transformer/kernels_transformer.cuh transformer/utils/checks.cuh transformer/kernels_transformer.cu transformer/FeedForward.cu transformer/FeedForward.cuh transformer/Decoder.cu transformer/Decoder.cuh transformer/Transformer.cu transformer/Transformer.cuh)

set_target_properties(Nlp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Nlp PROPERTIES CUDA_ARCHITECTURES "80;80")

target_include_directories(Nlp
        PUBLIC ${CUDA_TOOLKIT_PATH}/include
        )

target_link_libraries(Nlp
        PUBLIC cusparseLt
        PUBLIC cublas
        PUBLIC cusparse
        PUBLIC cudnn
        )