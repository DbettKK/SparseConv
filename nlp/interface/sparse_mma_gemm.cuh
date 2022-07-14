//
// Created by dbettkk on 2022/7/14.
//

#ifndef SPARSECONV_SPARSE_MMA_GEMM_CUH
#define SPARSECONV_SPARSE_MMA_GEMM_CUH

#include <iostream>
#include <string>
#include <fstream>

#include <cuda_fp16.h>
#include <cusparseLt.h>

using namespace std;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",          \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        return;                                                                \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at %s line %d with error: %s (%d)\n",      \
               __FILE__, __LINE__, cusparseGetErrorString(status), status);    \
        return;                                                                \
    }                                                                          \
}

void sparse_mma_gemm_device(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, bool isValid, half *outputD);

/**
 * 从 host 端向 device 端拷贝数据的同时进行 padding 操作
 * @param src 源矩阵
 * @param row 源矩阵行
 * @param col 源矩阵列
 * @param dest 目标矩阵
 * @param row_padding 目标矩阵行
 * @param col_padding 目标矩阵列
 */
void padCudaMemcpy2D(const half* src, int row, int col, half *dest, int row_padding, int col_padding);

#endif //SPARSECONV_SPARSE_MMA_GEMM_CUH
