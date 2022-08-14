cmake_minimum_required(VERSION 3.22)
project(Nlp CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(Nlp nlp_main.cu
        ../spmma/utils/CudaTime.cu ../spmma/utils/CudaTime.cuh
        transformer/Encoder.cu transformer/Encoder.cuh transformer/Attention.cu transformer/Attention.cuh transformer/MatrixHalf.cu transformer/MatrixHalf.cuh transformer/interface/kernels_transformer.cuh transformer/utils/checks.cuh transformer/interface/kernels_transformer.cu transformer/FeedForward.cu transformer/FeedForward.cuh transformer/Decoder.cu transformer/Decoder.cuh transformer/Transformer.cu transformer/Transformer.cuh test/test_nlp.cu test/test_nlp.cuh transformer/interface/spmma_interface.cu transformer/interface/spmma_interface.cuh transformer/interface/cublas_interface.cu transformer/interface/cublas_interface.cuh transformer/interface/cusparse_interface.cu transformer/interface/cusparse_interface.cuh test/test_spmma.cu test/test_spmma.cuh test/Test.cu test/Test.cuh)

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