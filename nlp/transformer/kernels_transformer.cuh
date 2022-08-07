//
// Created by dbettkk on 2022/7/24.
//

#ifndef SPARSECONV_KERNELS_TRANSFORMER_CUH
#define SPARSECONV_KERNELS_TRANSFORMER_CUH

#include<cuda_fp16.h>
#include <cudnn.h>
#include <iostream>
#include<cmath>
#include "utils/checks.cuh"
#include "../../spmma/utils/CudaTime.cuh"

__global__ void reshape_multi_head(half *A, half *B, int row, int col, int heads);

/**
 * 对每一列进行 softmax
 * 每一列对应一个block
 */
__global__ void softmax_half(half *item, int row, int col);

__global__ void transpose_half(half *item, half *out, int row, int col);

__global__ void gemm_simple(half *A, half *B, int m, int k, int n, half *out);

__global__ void mask_matrix_gpu(half *tgt, const int *mask_mat, int row, int col);

__global__ void relu_half(half *item, int row, int col);

__global__ void matrix_add(half *A, half *B, half *C, int batch, int A_row, int A_col, int B_row, int B_col);

__global__ void layerNorm_kernel(half *feature, int batch, int max_len, int size, half *means, half *std, half *out);

__global__ void getMeanAndStd(half *feature, int batch, int max_len, int size, half *means, half *std);

/* 得到的 output 为转置后的 */
void cublas_gemm_device(const half *d_A, const half *d_B, int inputM, int inputK, int inputN, half *output);

void cublas_gemm_device_scale(const half *d_A, const half *d_B, int inputM, int inputK, int inputN, float scale, half *output);

void cublas_gemm_batches_device(half *d_A, half *d_B, int batch, int inputM, int inputK, int inputN, bool isSingleBatch, half *output);

void padCudaMemcpy2D(const half* src, int row, int col, half *dest, int row_padding, int col_padding);

void sparse_mma_gemm_device(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, bool isValid, half *outputD);

void cusparse_gemm_csr_device(half *sp_A, half *d_B, int m, int k, int n, half *output);

void cusparse_gemm_blocked_device_test();

void cusparse_gemm_csr_device_test();

void softmax_cudnn_trans(half *feature, int batch, int channel, int width, int height, half *out);


#endif //SPARSECONV_KERNELS_TRANSFORMER_CUH
