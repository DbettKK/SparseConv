//
// Created by dbettkk on 2022/7/24.
//

#ifndef SPARSECONV_KERNELS_TRANSFORMER_CUH
#define SPARSECONV_KERNELS_TRANSFORMER_CUH

#include<cuda_fp16.h>
#include <cudnn.h>
#include <iostream>
#include<cmath>
#include "../utils/checks.cuh"
#include "../../../utils/CudaTime.cuh"

__global__ void reshape_multi_head(half *A, half *B, int row, int col, int heads);

/**
 * 对每一列进行 softmax
 * 每一列对应一个block
 */
__global__ void softmax_half(half *item, int row, int col, half *out);

__global__ void softmax_batches(half *item, const int batch, const int row, const int col, half *out);

__global__ void softmax_half_v2(half *item, const int row, const int col, half *out);

__global__ void transpose_half(half *item, half *out, int row, int col);

__global__ void transpose_batches(half *item, half *out, int batch, int row, int col);

__global__ void gemm_simple(half *A, half *B, int m, int k, int n, half *out);

__global__ void mask_matrix_gpu(half *tgt, const int *mask_mat, int row, int col);

__global__ void mask_matrix_batches(half *tgt, const int *mask_mat, int batch, int row, int col);

__global__ void relu_half(half *item, int size);

__global__ void matrix_add(half *A, half *B, half *C, int batch, int A_row, int A_col, int B_row, int B_col);

__global__ void layerNorm_kernel(half *feature, int batch, int max_len, int size, half *means, half *std, half *out);

__global__ void getMeanAndStd(half *feature, int batch, int max_len, int size, half *means, half *std);

void softmax_cudnn_trans(half *feature, int batch, int channel, int width, int height, half *out);


#endif //SPARSECONV_KERNELS_TRANSFORMER_CUH
