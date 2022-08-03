//
// Created by dbettkk on 2022/8/1.
//

#ifndef SPARSECONV_INTERFACES_CUH
#define SPARSECONV_INTERFACES_CUH

#include <cuda_fp16.h>
#include <cudnn.h>

#include "../interface/resnet_kernel.cuh"
#include "../../nlp/transformer/kernels_transformer.cuh"

#define CHECK_CUDNN(func)                                                      \
{                                                                              \
    cudnnStatus_t status = (func);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
        printf("CUDNN failed at line %d with error: %s (%d)\n",                \
               __LINE__, cudnnGetErrorString(status), status);                 \
        return ;                                                               \
    }                                                                          \
}

__global__ void ReLU(half *in, half *out, int size);


void conv2d_device_spmma(half *feature, half *kernel, int batch, int in_c, int out_c, int f_w, int f_h, int k_w, int k_h, int stride, int padding, half *out);

void conv2d_device_cudnn(half *feature, half *kernel, int batch, int in_c, int out_c, int f_w, int f_h, int k_w, int k_h, int stride, int padding, half *out);

void bn_cudnn(half *feature, int batch, int channel, int width, int height, half *out);

void pool_cudnn(half *feature, int batch, int channel, int width, int height, half *out, int window_size, int padding, int stride, int modes);

void softmax_cudnn(half *feature, int batch, int channel, int width, int height, half *out);

#endif //SPARSECONV_INTERFACES_CUH