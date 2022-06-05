#include<mma.h>
#include<cuda_fp16.h>
#include <iostream>


__global__ void __launch_bounds__(32) default_function_kernel0(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1]; // 声明结果 类
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1]; // 声明矩阵A 类
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1]; // 声明矩阵B 类
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f); // 赋值
    for (int ry = 0; ry < 4; ++ry) {
        for (int rx = 0; rx < 4; ++rx) {    // rx ry对应filter的h w
            __syncthreads();
            for (int i = 0; i < 8; ++i) {
                compute_shared[i * 32 + (int)threadIdx.x] =
                        Data[
                                i * 1568 +
                                ((int)threadIdx.x >> 4) * 784 +
                                ((int)blockIdx.z >> 2) * 112 +
                                ry * 112 +
                                rx * 16 +
                                ((int)blockIdx.z & 3) * 16 +
                                ((int)threadIdx.x & 15)];
            }
            for (int i = 0; i < 8; ++i) {
                compute_d_shared[i * 32 + (int)threadIdx.x] = Filter[ry * 1024 + rx * 256 + i * 32 + (int)threadIdx.x];
            }
            __syncthreads();
            (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((half *)compute_shared + (0)), 16);
            (void)nvcuda::wmma::load_matrix_sync(compute_d_shared_wmma_matrix_b[0], ((half *)compute_d_shared + (0)), 16);
            (void)nvcuda::wmma::mma_sync(Conv2dOutput_wmma_accumulator[0], compute_shared_wmma_matrix_a[0], compute_d_shared_wmma_matrix_b[0], Conv2dOutput_wmma_accumulator[0]);
        }
    }
    __syncthreads();
    (void)nvcuda::wmma::store_matrix_sync(((half *)compute_shared + (0)), Conv2dOutput_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (int i = 0; i < 8; ++i) {
        Conv2dOutput[i * 512 + ((int)threadIdx.x >> 4) * 256 + ((int)blockIdx.z) * 16 + ((int)threadIdx.x & 15)] =
                compute_shared[i * 32 + (int)threadIdx.x];
    }
}

extern "C" void conv2d(float *f_data, float *f_filter, float *f_out) {

    half *h_data = new half[256 * 49];
    half *h_filter = new half[16 * 256];
    half *h_out = new half[16 * 256];

    for (int i = 0; i < 256*49; i++) h_data[i] = __float2half(f_data[i]);
    for (int i = 0; i < 256*16; i++) h_filter[i] = __float2half(f_filter[i]);

    half *d_data, *d_filter, *d_out;
    cudaMalloc((void **)&d_data, sizeof(half) * 256*49);
    cudaMalloc((void **)&d_filter, sizeof(half) * 256*16);
    cudaMalloc((void **)&d_out, sizeof(half) * 256*16);

    cudaMemcpy(d_data, h_data, sizeof(half) * 256*49, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(half) * 256*16, cudaMemcpyHostToDevice);

    dim3 block(1, 1, 16);
    dim3 thread(32, 1, 1);
    default_function_kernel0<<<block,thread>>>(d_data, d_filter, d_out);

    cudaMemcpy(h_out, d_out, sizeof(half) * 256*16, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16 * 256; i++) {
        if ( __half2int_rz(h_out[i]) == 0)  printf("false");
        f_out[i] = __half2float(h_out[i]);
    }
}

int main() {
    half *h_data = new half[256 * 49];
    half *h_filter = new half[16 * 256];
    half *h_out = new half[16 * 256];

    for (int i = 0; i < 256*49; i++) h_data[i] = __float2half(1.0);
    for (int i = 0; i < 256*16; i++) h_filter[i] = __float2half(1.0);

    half *d_data, *d_filter, *d_out;
    cudaMalloc((void **)&d_data, sizeof(half) * 256*49);
    cudaMalloc((void **)&d_filter, sizeof(half) * 256*16);
    cudaMalloc((void **)&d_out, sizeof(half) * 256*16);

    cudaMemcpy(d_data, h_data, sizeof(half) * 256*49, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(half) * 256*16, cudaMemcpyHostToDevice);

    dim3 block(1, 1, 16);
    dim3 thread(32, 1, 1);
    default_function_kernel0<<<block,thread>>>(d_data, d_filter, d_out);

    cudaMemcpy(h_out, d_out, sizeof(half) * 256*16, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16 * 256; i++) {
        if ( __half2int_rz(h_out[i]) == 0)  printf("false");
    }
}

