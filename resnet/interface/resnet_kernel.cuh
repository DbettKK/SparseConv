//
// Created by dbettkk on 2022/3/31.
//

#ifndef SPARSECONVOLUTION_KERNELS_CUH
#define SPARSECONVOLUTION_KERNELS_CUH

#include<iostream>
#include <cuda_fp16.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",          \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        return;                                                                \
    }                                                                          \
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);  i += blockDim.x * gridDim.x)

#define CUDA_POST_KERNEL_CHECK CHECK_CUDA(cudaPeekAtLastError())

// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__global__ void im2col_gpu_kernel(int n, const half *data_im, int data_n, int channel,
                                  int height, int width, int kernel_h, int kernel_w,
                                  int pad_h, int pad_w,
                                  int stride_h, int stride_w,
                                  int dilation_h, int dilation_w,
                                  int height_col, int width_col, half *data_col);

void im2col_gpu(const half *data_im, int data_n, int channels,
                int height, int width, int kernel_h, int kernel_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                int dilation_h, int dilation_w, half *data_col);

template<typename Dtype>
__global__ void im2col_rev_kernel(
        const int n, const Dtype *data, int data_n, int kernel_n, int out_h, int out_w, Dtype *out) {
    // row: kernel_n
    // col: n * out_h * out_w
    // n * out_h * out_w个线程
    CUDA_KERNEL_LOOP(index, n) {
        // 每个thread负责一个卷积核对应位置的所有channel
        int line = index % (data_n * out_h * out_w);
        int n_index = index / (out_h * out_w);
        int h_index = (index - n_index * (out_h * out_w)) / out_w;
        int w_index = (index - n_index * (out_h * out_w)) - h_index * out_w;
        for (int i = 0; i < kernel_n; i++) {
            out[n_index * kernel_n * out_h * out_w + i * out_h * out_w + h_index * out_w + w_index] = data[
                    i * data_n * out_h * out_w + line];
            //half a = data[i * data_n * out_h * out_w + line];
            //out[n_index * kernel_n * out_h * out_w + i * out_h * out_w + h_index * out_w + w_index] = 0;
        }
    }
}

#endif //SPARSECONVOLUTION_KERNELS_CUH
