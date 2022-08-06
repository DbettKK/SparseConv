//
// Created by dbettkk on 2022/7/24.
//

#ifndef SPARSECONV_CHECKS_CUH
#define SPARSECONV_CHECKS_CUH

#include <cublas_v2.h>
#include <cusparse.h>
#include <cusparseLt.h>

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

#define CHECK_CUDNN(func)                                                      \
{                                                                              \
    cudnnStatus_t status = (func);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
        printf("CUDNN failed at line %d with error: %s (%d)\n",                \
               __LINE__, cudnnGetErrorString(status), status);                 \
        return ;                                                               \
    }                                                                          \
}

#endif //SPARSECONV_CHECKS_CUH
