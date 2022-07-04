//
// Created by dbettkk on 2022/6/11.
//

#ifndef SPARSECONV_WMMA_SP_CUH
#define SPARSECONV_WMMA_SP_CUH

#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <mma.h>
#include <driver_types.h>
#include "../spmma/utils/CudaTime.cuh"
#include "utils/DataGenerator.cuh"

using namespace nvcuda;
using namespace std;

#define CHECK_CUDA_ERROR(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",          \
               __FILE__, __LINE__, cudaGetErrorString(status), status);              \
        return;                                                                             \
    }                                                                          \
}

#endif //SPARSECONV_WMMA_SP_CUH
