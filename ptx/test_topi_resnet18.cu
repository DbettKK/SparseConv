//
// Created by dbettkk on 2022/7/4.
//
#include "wmma.sp.cuh"

__global__ void __launch_bounds__(32) resnet_conv2(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
    for (int ry = 0; ry < 3; ++ry) {
        for (int rx = 0; rx < 3; ++rx) {
            for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
                __syncthreads();
                for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 56) + ry)) && (((((int)blockIdx.z) / 56) + ry) < 57)) && (1 <= (rx + (((int)blockIdx.z) % 56)))) && ((rx + (((int)blockIdx.z) % 56)) < 57)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 401408) + ((((int)threadIdx.x) >> 4) * 200704)) + (ry * 3584)) + (((int)blockIdx.z) * 64)) + (rx * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3648))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 12288) + (rx * 4096)) + (rc_outer_outer * 1024)) + (ax2_ax3_fused_outer_outer_outer_outer * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 401408) + ((((int)threadIdx.x) >> 4) * 200704)) + (((int)blockIdx.z) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv3(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
    for (int ry = 0; ry < 3; ++ry) {
        for (int rx = 0; rx < 3; ++rx) {
            for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
                __syncthreads();
                for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 56) + ry)) && (((((int)blockIdx.z) / 56) + ry) < 57)) && (1 <= (rx + (((int)blockIdx.z) % 56)))) && ((rx + (((int)blockIdx.z) % 56)) < 57)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 401408) + ((((int)threadIdx.x) >> 4) * 200704)) + (ry * 3584)) + (((int)blockIdx.z) * 64)) + (rx * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3648))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 12288) + (rx * 4096)) + (rc_outer_outer * 1024)) + (ax2_ax3_fused_outer_outer_outer_outer * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 401408) + ((((int)threadIdx.x) >> 4) * 200704)) + (((int)blockIdx.z) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv4(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
    for (int ry = 0; ry < 3; ++ry) {
        for (int rx = 0; rx < 3; ++rx) {
            for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
                __syncthreads();
                for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 56) + ry)) && (((((int)blockIdx.z) / 56) + ry) < 57)) && (1 <= (rx + (((int)blockIdx.z) % 56)))) && ((rx + (((int)blockIdx.z) % 56)) < 57)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 401408) + ((((int)threadIdx.x) >> 4) * 200704)) + (ry * 3584)) + (((int)blockIdx.z) * 64)) + (rx * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3648))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 12288) + (rx * 4096)) + (rc_outer_outer * 1024)) + (ax2_ax3_fused_outer_outer_outer_outer * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 401408) + ((((int)threadIdx.x) >> 4) * 200704)) + (((int)blockIdx.z) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv5(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
    for (int ry = 0; ry < 3; ++ry) {
        for (int rx = 0; rx < 3; ++rx) {
            for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
                __syncthreads();
                for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 56) + ry)) && (((((int)blockIdx.z) / 56) + ry) < 57)) && (1 <= (rx + (((int)blockIdx.z) % 56)))) && ((rx + (((int)blockIdx.z) % 56)) < 57)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 401408) + ((((int)threadIdx.x) >> 4) * 200704)) + (ry * 3584)) + (((int)blockIdx.z) * 64)) + (rx * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3648))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 12288) + (rx * 4096)) + (rc_outer_outer * 1024)) + (ax2_ax3_fused_outer_outer_outer_outer * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 401408) + ((((int)threadIdx.x) >> 4) * 200704)) + (((int)blockIdx.z) * 64)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv6(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
    for (int ry = 0; ry < 3; ++ry) {
        for (int rx = 0; rx < 3; ++rx) {
            for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
                __syncthreads();
                for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((1 <= (((((int)blockIdx.z) / 28) * 2) + ry)) && (1 <= (((((int)blockIdx.z) % 28) * 2) + rx))) ? Data[((((((((((ax0_ax3_fused_outer_outer_outer_outer * 401408) + ((((int)threadIdx.x) >> 4) * 200704)) + ((((int)blockIdx.z) / 28) * 7168)) + (ry * 3584)) + ((((int)blockIdx.z) % 28) * 128)) + (rx * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3648))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 24576) + (rx * 8192)) + (rc_outer_outer * 2048)) + (ax2_ax3_fused_outer_outer_outer_outer * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 200704) + ((((int)threadIdx.x) >> 4) * 100352)) + (((int)blockIdx.z) * 128)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv7(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
    for (int ry = 0; ry < 3; ++ry) {
        for (int rx = 0; rx < 3; ++rx) {
            for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
                __syncthreads();
                for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 28) + ry)) && (((((int)blockIdx.z) / 28) + ry) < 29)) && (1 <= (rx + (((int)blockIdx.z) % 28)))) && ((rx + (((int)blockIdx.z) % 28)) < 29)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 200704) + ((((int)threadIdx.x) >> 4) * 100352)) + (ry * 3584)) + (((int)blockIdx.z) * 128)) + (rx * 128)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3712))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 49152) + (rx * 16384)) + (rc_outer_outer * 2048)) + (ax2_ax3_fused_outer_outer_outer_outer * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 200704) + ((((int)threadIdx.x) >> 4) * 100352)) + (((int)blockIdx.z) * 128)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv8(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
    for (int ry = 0; ry < 3; ++ry) {
        for (int rx = 0; rx < 3; ++rx) {
            for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
                __syncthreads();
                for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 28) + ry)) && (((((int)blockIdx.z) / 28) + ry) < 29)) && (1 <= (rx + (((int)blockIdx.z) % 28)))) && ((rx + (((int)blockIdx.z) % 28)) < 29)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 200704) + ((((int)threadIdx.x) >> 4) * 100352)) + (ry * 3584)) + (((int)blockIdx.z) * 128)) + (rx * 128)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3712))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 49152) + (rx * 16384)) + (rc_outer_outer * 2048)) + (ax2_ax3_fused_outer_outer_outer_outer * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 200704) + ((((int)threadIdx.x) >> 4) * 100352)) + (((int)blockIdx.z) * 128)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv9(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
    for (int ry = 0; ry < 3; ++ry) {
        for (int rx = 0; rx < 3; ++rx) {
            for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
                __syncthreads();
                for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 28) + ry)) && (((((int)blockIdx.z) / 28) + ry) < 29)) && (1 <= (rx + (((int)blockIdx.z) % 28)))) && ((rx + (((int)blockIdx.z) % 28)) < 29)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 200704) + ((((int)threadIdx.x) >> 4) * 100352)) + (ry * 3584)) + (((int)blockIdx.z) * 128)) + (rx * 128)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3712))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 49152) + (rx * 16384)) + (rc_outer_outer * 2048)) + (ax2_ax3_fused_outer_outer_outer_outer * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 200704) + ((((int)threadIdx.x) >> 4) * 100352)) + (((int)blockIdx.z) * 128)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv10(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
    for (int ry = 0; ry < 3; ++ry) {
        for (int rx = 0; rx < 3; ++rx) {
            for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
                __syncthreads();
                for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((1 <= (((((int)blockIdx.z) / 14) * 2) + ry)) && (1 <= (((((int)blockIdx.z) % 14) * 2) + rx))) ? Data[((((((((((ax0_ax3_fused_outer_outer_outer_outer * 200704) + ((((int)threadIdx.x) >> 4) * 100352)) + ((((int)blockIdx.z) / 14) * 7168)) + (ry * 3584)) + ((((int)blockIdx.z) % 14) * 256)) + (rx * 128)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3712))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 98304) + (rx * 32768)) + (rc_outer_outer * 4096)) + (ax2_ax3_fused_outer_outer_outer_outer * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + (((int)blockIdx.z) * 256)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv11(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
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
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 14) + ry)) && (((((int)blockIdx.z) / 14) + ry) < 15)) && (1 <= (rx + (((int)blockIdx.z) % 14)))) && ((rx + (((int)blockIdx.z) % 14)) < 15)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + (ry * 3584)) + (((int)blockIdx.z) * 256)) + (rx * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3840))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 196608) + (rx * 65536)) + (rc_outer_outer * 4096)) + (ax2_ax3_fused_outer_outer_outer_outer * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + (((int)blockIdx.z) * 256)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv12(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
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
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 14) + ry)) && (((((int)blockIdx.z) / 14) + ry) < 15)) && (1 <= (rx + (((int)blockIdx.z) % 14)))) && ((rx + (((int)blockIdx.z) % 14)) < 15)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + (ry * 3584)) + (((int)blockIdx.z) * 256)) + (rx * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3840))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 196608) + (rx * 65536)) + (rc_outer_outer * 4096)) + (ax2_ax3_fused_outer_outer_outer_outer * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + (((int)blockIdx.z) * 256)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv13(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
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
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 14) + ry)) && (((((int)blockIdx.z) / 14) + ry) < 15)) && (1 <= (rx + (((int)blockIdx.z) % 14)))) && ((rx + (((int)blockIdx.z) % 14)) < 15)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + (ry * 3584)) + (((int)blockIdx.z) * 256)) + (rx * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3840))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 196608) + (rx * 65536)) + (rc_outer_outer * 4096)) + (ax2_ax3_fused_outer_outer_outer_outer * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + (((int)blockIdx.z) * 256)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv14(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
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
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((1 <= (((((int)blockIdx.z) / 7) * 2) + ry)) && (1 <= (((((int)blockIdx.z) % 7) * 2) + rx))) ? Data[((((((((((ax0_ax3_fused_outer_outer_outer_outer * 100352) + ((((int)threadIdx.x) >> 4) * 50176)) + ((((int)blockIdx.z) / 7) * 7168)) + (ry * 3584)) + ((((int)blockIdx.z) % 7) * 512)) + (rx * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3840))] : __float2half_rn(0.000000e+00f));
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 50176) + ((((int)threadIdx.x) >> 4) * 25088)) + (((int)blockIdx.z) * 512)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv15(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
    for (int ry = 0; ry < 3; ++ry) {
        for (int rx = 0; rx < 3; ++rx) {
            for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
                __syncthreads();
                for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 7) + ry)) && (((((int)blockIdx.z) / 7) + ry) < 8)) && (1 <= (rx + (((int)blockIdx.z) % 7)))) && ((rx + (((int)blockIdx.z) % 7)) < 8)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 50176) + ((((int)threadIdx.x) >> 4) * 25088)) + (ry * 3584)) + (((int)blockIdx.z) * 512)) + (rx * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 4096))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 786432) + (rx * 262144)) + (rc_outer_outer * 8192)) + (ax2_ax3_fused_outer_outer_outer_outer * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 50176) + ((((int)threadIdx.x) >> 4) * 25088)) + (((int)blockIdx.z) * 512)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv16(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
    for (int ry = 0; ry < 3; ++ry) {
        for (int rx = 0; rx < 3; ++rx) {
            for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
                __syncthreads();
                for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 7) + ry)) && (((((int)blockIdx.z) / 7) + ry) < 8)) && (1 <= (rx + (((int)blockIdx.z) % 7)))) && ((rx + (((int)blockIdx.z) % 7)) < 8)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 50176) + ((((int)threadIdx.x) >> 4) * 25088)) + (ry * 3584)) + (((int)blockIdx.z) * 512)) + (rx * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 4096))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 786432) + (rx * 262144)) + (rc_outer_outer * 8192)) + (ax2_ax3_fused_outer_outer_outer_outer * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 50176) + ((((int)threadIdx.x) >> 4) * 25088)) + (((int)blockIdx.z) * 512)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}
__global__ void __launch_bounds__(32) resnet_conv17(half* __restrict__ Data, half* __restrict__ Filter, half* __restrict__ Conv2dOutput) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> Conv2dOutput_wmma_accumulator[1];
    __shared__ half compute_shared[256];
    __shared__ half compute_d_shared[256];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> compute_d_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(Conv2dOutput_wmma_accumulator[0], 0.000000e+00f);
    for (int ry = 0; ry < 3; ++ry) {
        for (int rx = 0; rx < 3; ++rx) {
            for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
                __syncthreads();
                for (int ax0_ax3_fused_outer_outer_outer_outer = 0; ax0_ax3_fused_outer_outer_outer_outer < 8; ++ax0_ax3_fused_outer_outer_outer_outer) {
                    compute_shared[(((ax0_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = (((((1 <= ((((int)blockIdx.z) / 7) + ry)) && (((((int)blockIdx.z) / 7) + ry) < 8)) && (1 <= (rx + (((int)blockIdx.z) % 7)))) && ((rx + (((int)blockIdx.z) % 7)) < 8)) ? Data[(((((((((ax0_ax3_fused_outer_outer_outer_outer * 50176) + ((((int)threadIdx.x) >> 4) * 25088)) + (ry * 3584)) + (((int)blockIdx.z) * 512)) + (rx * 512)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 4096))] : __float2half_rn(0.000000e+00f));
                }
                for (int ax2_ax3_fused_outer_outer_outer_outer = 0; ax2_ax3_fused_outer_outer_outer_outer < 8; ++ax2_ax3_fused_outer_outer_outer_outer) {
                    compute_d_shared[(((ax2_ax3_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))] = Filter[((((((((ry * 786432) + (rx * 262144)) + (rc_outer_outer * 8192)) + (ax2_ax3_fused_outer_outer_outer_outer * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))];
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
        Conv2dOutput[((((((nn_inner_ff_inner_fused_outer_outer_outer_outer * 50176) + ((((int)threadIdx.x) >> 4) * 25088)) + (((int)blockIdx.z) * 512)) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) & 15)))] = compute_shared[(((nn_inner_ff_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x)))];
    }
}


void read_bin(half *item, int size, string path) {
    std::ifstream in(path, std::ios::binary);
    in.read((char *)item, size * sizeof(half));
    in.close();
}

void test_conv(int data_size, int filter_size, int out_size, string d_p, string f_p) {
    half *hData = new half[data_size];
    half *hFilter = new half[filter_size];
    half *hOut = new half[out_size];

    read_bin(hData, data_size, d_p);
    read_bin(hFilter, filter_size, f_p);

    half *dData, *dFilter, *dOut;
    cudaMalloc(&dData, sizeof(half) * data_size);
    cudaMalloc(&dFilter, sizeof(half) * filter_size);
    cudaMalloc(&dOut, sizeof(half) * out_size);
    cudaMemcpy(dData, hData, sizeof(half) * data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dFilter, hFilter, sizeof(half) * filter_size, cudaMemcpyHostToDevice);

    dim3 grid(1, 1, 1);
    dim3 block(1, 1, 1);

    CudaTime *time = new CudaTime();
    time->initAndStart();

    resnet_conv10<<<grid, block>>>(dData, dFilter, dOut);

    float t = time->endAndGetTime();
    printf("%f\n", t);

    cudaMemcpy(hOut, dOut, sizeof(half) * out_size, cudaMemcpyDeviceToHost);
    // do sth
    cudaFree(dData);
    cudaFree(dFilter);
    cudaFree(dOut);
    delete[] hData;
    delete[] hFilter;
    delete[] hOut;
}