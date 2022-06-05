//
// Created by dbettkk on 2022/3/30.
//

#ifndef SPARSECONVOLUTION_UTIL_CUH
#define SPARSECONVOLUTION_UTIL_CUH

#include<iostream>
#include<cuda_fp16.h>
#include<cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include<cusparseLt.h>       // cusparseLt header

using namespace std;

// without ret
#define CUDA_CHECK(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",          \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        return nullptr;                                                        \
    }                                                                          \
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",          \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        return;                                                                \
    }                                                                          \
}

#define CHECK_CUDA_NO_RET(func)                                                \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",          \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
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

template <typename Dtype>
void decimal2binary(Dtype num, int byteNum) {
    int *bottle = new int[byteNum];
    for (int i = 0; i < byteNum; i++) {
        bottle[i] = num & 1;
        num = num >> 1;

    }
    for (int i = byteNum - 1; i >= 0; i--) {
        if ((i + 1) % 4 == 0) printf(" ");
        printf("%d", bottle[i]);
    }
    delete[] bottle;
}

short convertIdx2Binary(int *index, int len);

size_t get_cmpr_size(int row, int col);

bool checkGPU();

void float2half_array(float *in, half *out, int totalSize);

void half2float_array(half *in, float *out, int totalSize);

template <typename Dtype>
void padMatrix(const Dtype* src, int row, int col, Dtype *dest, int row_padding, int col_padding) {
    CHECK_CUDA( cudaMemset(dest, 0, row_padding * col_padding * sizeof(Dtype)) )
    if (col == col_padding) {
        CHECK_CUDA( cudaMemcpy(dest, src, row * col_padding * sizeof(Dtype), cudaMemcpyDeviceToDevice) )
    } else {
        // spitch指定想要复制的矩阵的本身的宽度 width指定需要复制的宽度 dpitch指定赋值到dest的宽度
        CHECK_CUDA( cudaMemcpy2D(dest, col_padding * sizeof(Dtype), src, col * sizeof(Dtype), col * sizeof(Dtype), row, cudaMemcpyDeviceToDevice) )
    }
}

template <typename Dtype>
void restoreMatrix(const Dtype* src, int row, int col, Dtype *dest, int row_restore, int col_restore, bool toDevice) {
    auto direction = toDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    if (col == col_restore) {
        CHECK_CUDA( cudaMemcpy(dest, src, row_restore * col * sizeof(Dtype), direction) )
    } else {
        CHECK_CUDA(cudaMemcpy2D(dest, col_restore * sizeof(Dtype), src, col * sizeof(Dtype), col_restore * sizeof(Dtype), row_restore, direction) )
    }
}

template <typename Dtype>
Dtype* transpose(Dtype *item, int row, int col) {
    // row col是转置前的
    auto *ret = new Dtype[row * col];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            ret[j * row + i] = item[i * col + j];
        }
    }
    return ret;
}

template <typename Dtype>
bool cmpMatrix(Dtype *item1, Dtype *item2, int totalSize) {
    for (int i = 0; i < totalSize; i++) {
        if (__half2float(item1[i]) != __half2float(item2[i])) {
            return false;
        }
    }
    return true;
}

template <typename Dtype>
bool check_sparse(Dtype *item, int row, int col) {
    printf("m: %d, k: %d\n", row, col);
    Dtype *host = new Dtype[row * col];
    cudaMemcpy(host, item, sizeof(Dtype) * row * col, cudaMemcpyDeviceToHost);
//    printf("padding: \n");
//    for (int i = 0; i < row; i++) {
//        for (int j = 0; j < col; j++) {
//            printf("%d ", __half2int_rz(host[i * col + j]));
//        }
//        printf("\n");
//    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j+=4) {
            int zero_cnt = 0;
            for (int start = 0; start < 4; start++) {
                if (__half2float(host[i * col + j + start]) == 0) {
                    zero_cnt++;
                }
            }
            if (zero_cnt < 2) return false;
        }
    }
    return true;
}

#endif //SPARSECONVOLUTION_UTIL_CUH
