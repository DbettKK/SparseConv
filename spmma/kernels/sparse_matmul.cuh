//
// Created by dbettkk on 2022/3/30.
//

#ifndef SPARSECONVOLUTION_SPARSE_MATMUL_CUH
#define SPARSECONVOLUTION_SPARSE_MATMUL_CUH

#include<cusparseLt.h>
#include<cuda_fp16.h>
#include<cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include<string>

#include "../entity/ConvParam.cuh"
#include "../utils/CudaTime.cuh"

/*
    src: device IN
    dest: device OUT (need to allocate)
*/
void spmma_matmul(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, bool isValid, half *outputD, MatrixParam *retParam);

#endif //SPARSECONVOLUTION_SPARSE_MATMUL_CUH
