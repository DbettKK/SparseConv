//
// Created by dbettkk on 2022/7/25.
//
#include "kernels_transformer.cuh"

bool check_sparse(half *item, int row, int col) {
    printf("m: %d, k: %d\n", row, col);
    half *host = new half[row * col];
    cudaMemcpy(host, item, sizeof(half) * row * col, cudaMemcpyDeviceToHost);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j+=4) {
            int zero_cnt = 0;
            for (int start = 0; start < 4; start++) {
                if (__half2float(host[i * col + j + start]) == 0) {
                    zero_cnt++;
                }
            }
            if (zero_cnt < 2) {
                return false;
            }
        }
    }
    return true;
}

__global__ void softmax_half(half *item, const int row, const int col, half *out) {
    // todo: row, col > 256怎么办
//    __shared__ float mem[64][64]; // 记录
//    const int blx = blockIdx.x;  // row
//    const int thx = threadIdx.x; // col
//    if (thx < col && blx < row) {
//        mem[blx][thx] = __half2float(item[blx * col + thx]); // 传入
//    }
//    __syncthreads();
//    if (thx < col && blx < row) {
//        float max = -65504;
//        for (int i = 0; i < col; i++) {
//            if (max <= mem[blx][i]) max = mem[blx][i];
//        }
//        double sum = 0;
//        for (int i = 0; i < col; i++) {
//            sum += expf(mem[blx][i] - max);
//        }
//        out[blx * col + thx] = expf(mem[blx][thx] - max) / sum;
//    }
    __shared__ float mem[513];
    const int blx = blockIdx.x;  // row
    const int thx = threadIdx.x; // col
    if (thx < col && blx < row) {
        mem[thx] = __half2float(item[blx * col + thx]);
    }
    __syncthreads();
    if (thx < col && blx < row) {
        float max = -65504;
        for (int i = 0; i < col; i++) {
            if (max <= mem[i]) max = mem[i];
        }
        double sum = 0;
        for (int i = 0; i < col; i++) {
            sum += expf(mem[i] - max);
        }
        out[blx * col + thx] = expf(mem[thx] - max) / sum;
    }
}

__global__ void softmax_half_v2(half *item, const int row, const int col, half *out) {
    __shared__ float mem[513], max_s;
    __shared__ double sum_s;
    const int blx = blockIdx.x;  // row
    const int thx = threadIdx.x; // col
    if (thx < col && blx < row) {
        mem[thx] = __half2float(item[blx * col + thx]);
    }
    __syncthreads();
    if (thx < col && blx < row) {
        if (thx == 0) {
            float max = -65504;
            for (int i = 0; i < col; i++) {
                if (max <= mem[i]) max = mem[i];
            }
            double sum = 0;
            for (int i = 0; i < col; i++) {
                sum += expf(mem[i] - max);
            }
            sum_s = sum;
            max_s = max;
        }
        __syncthreads();
        out[blx * col + thx] = expf(mem[thx] - max_s) / sum_s;
    }
}

__global__ void softmax_batches(half *item, const int batch, const int row, const int col, half *out) {
    __shared__ float mem[513], max_s;
    __shared__ double sum_s;
    const int blx = blockIdx.x;  // row
    const int batch_id = blockIdx.y;
    const int thx = threadIdx.x; // col
    if (thx < col && blx < row) {
        mem[thx] = __half2float(item[batch_id * row * col + blx * col + thx]);
    }
    __syncthreads();
    if (thx < col && blx < row) {
        if (thx == 0) {
            float max = -65504;
            for (int i = 0; i < col; i++) {
                if (max <= mem[i]) max = mem[i];
            }
            double sum = 0;
            for (int i = 0; i < col; i++) {
                sum += expf(mem[i] - max);
            }
            sum_s = sum;
            max_s = max;
        }
        __syncthreads();
        out[batch_id * row * col + blx * col + thx] = expf(mem[thx] - max_s) / sum_s;
    }
}

__global__ void reshape_multi_head(half *A, half *B, const int row, const int col, const int heads)
{
    const int thx = threadIdx.x;
    const int blx = blockIdx.x, bly = blockIdx.y, blz = blockIdx.z;
    B[blx * row * col + (bly * row + blz) * col / heads + thx] = A[blx * row * col + blz * col + (bly * col / heads + thx)];
}

__global__ void transpose_half(half *item, half *out, int row, int col) {
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < row && ny < col)
    {
        out[ny * row + nx] = item[nx * col + ny];
    }
}

__global__ void transpose_batches(half *item, half *out, int batch, int row, int col) {
    int blx = blockIdx.x, row_id = blockIdx.y, col_id = blockIdx.z;
    int batch_id = blx * 32 + threadIdx.x;
    if (batch_id < batch && row_id < row && col_id < col)
    {
        out[batch_id * row * col + col_id * row + row_id] = item[batch_id * row * col + row_id * col + col_id];
    }
}

__global__ void gemm_simple(half *A, half *B, int m, int k, int n, half *out) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < m && col < k) {
        for (int i = 0; i < n; i++) {
            out[row * k + col] += A[row * n + i] * B[col + i * k];
        }
    }
}

__global__ void mask_matrix_gpu(half *tgt, const int *mask_mat, int row, int col) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < row * col) {
        if (mask_mat[idx] == 0) tgt[idx] = -40000;
    }
}

__global__ void mask_matrix_batches(half *tgt, const int *mask_mat, int batch, int row, int col) {
    int batch_id = blockIdx.x * 32 + threadIdx.x, row_id = blockIdx.y, col_id = blockIdx.z;
    int idx = row_id * col + col_id;
    if (batch_id < batch) {
        if (mask_mat[idx] == 0) tgt[idx + batch_id * row * col] = -40000;
    }
}

__global__ void relu_half(half *item, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        if (item[idx] <= __float2half(0))
            item[idx] = 0;
    }
}

__global__ void matrix_add(half *A, half *B, half *C, int batch, int A_row, int A_col, int B_row, int B_col) {
    int batch_id = blockIdx.x, col_id = blockIdx.y;
    int row_id = threadIdx.x;
    int A_id = batch_id * A_row * A_col + row_id * A_col + col_id;
    int B_id = batch_id * B_row * B_col + row_id * B_col + col_id;
    C[A_id] = __half2float(A[A_id]) + __half2float(B[B_id]);
}

__global__ void layerNorm_kernel(half *feature, int batch, int max_len, int size, half *means, half *std, half *out) {
    int blx = blockIdx.x, thx = threadIdx.x;
    out[blx * size + thx] = (float)(feature[blx * size + thx] - means[blx]) / ((float)std[blx] + 0.01);
}

__global__ void getMeanAndStd(half *feature, int batch, int max_len, int size, half *means, half *std) {
    // 每个线程处理 size个元素
    int blx = blockIdx.x;
    int thx = threadIdx.x;
    float mean = 0, mean_2 = 0;
    for (int i = 0; i < size; i++) {
        float item = __half2float(feature[blx * max_len * size + thx * size + i]);
        mean += item / size;
        mean_2 += item / size * item;
    }
    means[blx * max_len + thx] = mean;
    float var = mean_2 - mean * mean;
    std[blx * max_len + thx] = sqrt(var > 0 ? var : 0);
}


void softmax_cudnn_trans(half *feature, int batch, int channel, int width, int height, half *out) {
    // handle
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle))

    float alpha = 1.0, beta = 0.0;

    // input
    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           batch, channel, height, width)) // n, c, h,
    // output
    cudnnTensorDescriptor_t output_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           batch, channel, height, width))

    // softmax
    CHECK_CUDNN(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
                                    input_descriptor, feature, &beta, output_descriptor, out))

    // free
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor))
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_descriptor))

    CHECK_CUDNN(cudnnDestroy(handle))
}

