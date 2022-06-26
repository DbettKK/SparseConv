#include<mma.h>
#include<cuda_fp16.h>
#include <iostream>
#include "utils/CudaTime.cuh"


__global__ void __launch_bounds__(32) default_function_kernel0(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
__shared__ half compute_shared[256];
__shared__ half compute_d_shared[256];
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
(void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
for (int ry = 0; ry < 3; ++ry) {
for (int rx = 0; rx < 3; ++rx) {
for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
__syncthreads();
for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 2097152) + ((((int)threadIdx.x) >> 4) * 1048576)) + ((((int)blockIdx.z) / 31) * 32768)) + (ry * 16384)) + ((((int)blockIdx.z) % 31) * 512)) + (rx * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)))];
}
for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 393216) + (rx * 131072)) + (rc_outer_outer * 8192)) + (ax2_ax3_fused_outer_outer_outer_outer * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
}
__syncthreads();
(void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((half *)compute_shared + (0)), 16);
(void)nvcuda::wmma::load_matrix_sync(compute_d_shared_wmma_matrix_b[0], ((half *)compute_d_shared + (0)), 16);
(void)nvcuda::wmma::mma_sync(Conv2dOutput_wmma_accumulator[0], compute_shared_wmma_matrix_a[0], compute_d_shared_wmma_matrix_b[0], Conv2dOutput_wmma_accumulator[0]);
}
}
}
__syncthreads();
(void)nvcuda::wmma::store_matrix_sync(((half *)compute_shared + (0)), Conv2dOutput_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
__syncthreads();
for (int nn_inner_ff_inner_fused_outer_outer_outer_outer = 0; nn_inner_ff_inner_fused_outer_outer_outer_outer < 8; ++nn_inner_ff_inner_fused_outer_outer_outer_outer) {
Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 984064) + ((((int)threadIdx.x) >> 4) * 492032)) + (((int)blockIdx.z) * 512)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
}
}

extern "C" void conv2d(float *f_data, float *f_filter, float *f_out) {

    half *h_data = new half[256 * 49];
    half *h_filter = new half[16 * 256];
    half *h_out = new half[16 * 256];

    for (int i = 0; i < 256 * 49; i++) h_data[i] = __float2half(f_data[i]);
    for (int i = 0; i < 256 * 16; i++) h_filter[i] = __float2half(f_filter[i]);

    half *d_data, *d_filter, *d_out;
    cudaMalloc((void **)&d_data, sizeof(half) * 256 * 49);
    cudaMalloc((void **)&d_filter, sizeof(half) * 256 * 16);
    cudaMalloc((void **)&d_out, sizeof(half) * 256 * 16);

    cudaMemcpy(d_data, h_data, sizeof(half) * 256 * 49, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(half) * 256 * 16, cudaMemcpyHostToDevice);

    dim3 block(1, 1, 16);
    dim3 thread(32, 1, 1);
    default_function_kernel0<<<block,thread>>>(d_data, d_filter, d_out);

    cudaMemcpy(h_out, d_out, sizeof(half) * 256 * 16, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16 * 256; i++) {
        if ( __half2int_rz(h_out[i]) == 0)  printf("false");
        f_out[i] = __half2float(h_out[i]);
    }
}

int main8() {
    half *h_data = new half[64*64*16*256];
    half *h_filter = new half[9*256*512];
    half *h_out = new half[16*512*32*32];

    for (int i = 0; i < 256*49; i++) h_data[i] = __float2half(1.0);
    for (int i = 0; i < 256*16; i++) h_filter[i] = __float2half(1.0);

    half *d_data, *d_filter, *d_out;
    cudaMalloc((void **)&d_data, sizeof(half) * 64*64*16*256);
    cudaMalloc((void **)&d_filter, sizeof(half) * 9*256*512);
    cudaMalloc((void **)&d_out, sizeof(half) * 16*512*32*32);

    cudaMemcpy(d_data, h_data, sizeof(half) * 64*64*16*256, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(half) * 9*256*512, cudaMemcpyHostToDevice);

    dim3 block(1, 32, 961);
    dim3 thread(32, 1, 1);

    for (int i = 0; i < 10; i++) {
        CudaTime *time = new CudaTime();
        time->initAndStart();

        default_function_kernel0<<<block,thread>>>(d_data, d_filter, d_out);

        float ti = time->endAndGetTime();
        printf("%fms\n", ti);
    }
    cudaMemcpy(h_out, d_out, sizeof(half) * 16*512*32*32, cudaMemcpyDeviceToHost);

//    for (int i = 0; i < 16 * 256; i++) {
//        if ( __half2int_rz(h_out[i]) != 0)  printf("%d ", __half2int_rz(h_out[i]));
//    }
    return 0;
}

