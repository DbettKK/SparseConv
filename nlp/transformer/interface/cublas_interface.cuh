//
// Created by dbettkk on 2022/8/14.
//

#ifndef SPARSECONV_CUBLAS_INTERFACE_CUH
#define SPARSECONV_CUBLAS_INTERFACE_CUH

#include "kernels_transformer.cuh"

void cublas_gemm_device(const half *d_A, const half *d_B, int inputM, int inputK, int inputN, half *output);

void cublas_gemm_device_scale(const half *d_A, const half *d_B, int inputM, int inputK, int inputN, float scale, half *output);

void cublas_gemm_batches_device(half *d_A, half *d_B, int batch, int inputM, int inputK, int inputN, bool isSingleBatch, half *output);

void cublas_gemm_batches_device_v2(half *d_A, half *d_B, int batch, int inputM, int inputK, int inputN, bool isSingleBatch, half *output);

void cublas_gemm_batches_scale_device(half *d_A, half *d_B, int batch, int inputM, int inputK, int inputN, float scale, half *output);

void cublas_gemm_batches_scale_device_v2(half *d_A, half *d_B, int batch, int inputM, int inputK, int inputN, float scale, half *output);

#endif //SPARSECONV_CUBLAS_INTERFACE_CUH
