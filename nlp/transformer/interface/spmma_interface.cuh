//
// Created by dbettkk on 2022/8/14.
//

#ifndef SPARSECONV_SPMMA_INTERFACE_CUH
#define SPARSECONV_SPMMA_INTERFACE_CUH

#include "kernels_transformer.cuh"

void padCudaMemcpy2D(const half* src, int row, int col, half *dest, int row_padding, int col_padding);

void sparse_mma_gemm_device(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, bool isValid, half *outputD);

void sparse_mma_gemm_noPad_device(half *inputA, half *inputB, int inputM, int inputK, int inputN, bool isValid, half *outputD);

void sparse_mma_gemm_batches_device(const half *inputA, const half *inputB, int batch, int inputM, int inputK, int inputN, bool isValid, half *outputD);

void sparse_mma_gemm_noPad_batches_device(half *inputA, half *inputB, int batch, int inputM, int inputK, int inputN, bool isValid, half *outputD);

void sparse_mma_gemm_device_v2(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, bool isValid, half *outputD);

void sparse_mma_gemm_splitK_device(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, bool isValid, half *outputD);

#endif //SPARSECONV_SPMMA_INTERFACE_CUH
