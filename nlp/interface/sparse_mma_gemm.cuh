//
// Created by dbettkk on 2022/7/14.
//

#ifndef SPARSECONV_SPARSE_MMA_GEMM_CUH
#define SPARSECONV_SPARSE_MMA_GEMM_CUH

#include <iostream>
#include <string>
#include <fstream>
#include <random>

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cusparseLt.h>

#include "../../spmma/utils/CudaTime.cuh"

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

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at %s line %d with error:  (%d)\n",          \
           __FILE__, __LINE__, status);                                        \
        return ;                                                               \
    }                                                                          \
}

void sparse_mma_gemm_device(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, bool isValid, half *outputD);

void cublas_gemm(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, half *output);
void cublas_gemm_device(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, half *output);

//void mask_gemm(half *inputA, half *inputB, int *mask, int inputM, int inputK, int inputN, half *output);
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

__global__ void transpose(half *src, half* tgt, int row, int col);

__global__ void mask_matrix_gpu(half *tgt, const int *mask_mat, int row, int col);

void position_encoding(half *input, int batch, int max_sen_len, int ebd);

void softmax(half *item);

#endif //SPARSECONV_SPARSE_MMA_GEMM_CUH
