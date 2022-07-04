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

void test_conv(int data_size, int filter_size, int out_size, string d_p, string f_p, int layer, int blockY, int blockZ) {
    CudaTime *time_all = new CudaTime();
    time_all->initAndStart();
    half *hData = new half[data_size];
    half *hFilter = new half[filter_size];
    half *hOut = new half[out_size];

    read_bin(hData, data_size, d_p);
    read_bin(hFilter, filter_size, f_p);

    half *dData, *dFilter, *dOut;
    CHECK_CUDA_ERROR(cudaMalloc(&dData, sizeof(half) * data_size));
    CHECK_CUDA_ERROR(cudaMalloc(&dFilter, sizeof(half) * filter_size));
    CHECK_CUDA_ERROR(cudaMalloc(&dOut, sizeof(half) * out_size));
    CHECK_CUDA_ERROR(cudaMemcpy(dData, hData, sizeof(half) * data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dFilter, hFilter, sizeof(half) * filter_size, cudaMemcpyHostToDevice));

    dim3 block(1, blockY, blockZ);
    dim3 thread(32, 1, 1);

    CudaTime *time = new CudaTime();
    time->initAndStart();

    switch (layer) {
        case 2: resnet_conv2<<<block, thread>>>(dData, dFilter, dOut); break;
        case 3: resnet_conv3<<<block, thread>>>(dData, dFilter, dOut); break;
        case 4: resnet_conv4<<<block, thread>>>(dData, dFilter, dOut); break;
        case 5: resnet_conv5<<<block, thread>>>(dData, dFilter, dOut); break;
        case 6: resnet_conv6<<<block, thread>>>(dData, dFilter, dOut); break;
        case 7: resnet_conv7<<<block, thread>>>(dData, dFilter, dOut); break;
        case 8: resnet_conv8<<<block, thread>>>(dData, dFilter, dOut); break;
        case 9: resnet_conv9<<<block, thread>>>(dData, dFilter, dOut); break;
        case 10: resnet_conv10<<<block, thread>>>(dData, dFilter, dOut); break;
        case 11: resnet_conv11<<<block, thread>>>(dData, dFilter, dOut); break;
        case 12: resnet_conv12<<<block, thread>>>(dData, dFilter, dOut); break;
        case 13: resnet_conv13<<<block, thread>>>(dData, dFilter, dOut); break;
        case 14: resnet_conv14<<<block, thread>>>(dData, dFilter, dOut); break;
        case 15: resnet_conv15<<<block, thread>>>(dData, dFilter, dOut); break;
        case 16: resnet_conv16<<<block, thread>>>(dData, dFilter, dOut); break;
        case 17: resnet_conv17<<<block, thread>>>(dData, dFilter, dOut); break;
    }

    float t = time->endAndGetTime();
    printf("%f\n", t);

    CHECK_CUDA_ERROR(cudaMemcpy(hOut, dOut, sizeof(half) * out_size, cudaMemcpyDeviceToHost));
    // do sth
    CHECK_CUDA_ERROR(cudaFree(dData));
    CHECK_CUDA_ERROR(cudaFree(dFilter));
    CHECK_CUDA_ERROR(cudaFree(dOut));
    delete[] hData;
    delete[] hFilter;
    delete[] hOut;

    float all_t = time_all->endAndGetTime();
    printf("all: %fms\n", all_t);
}

void test_resnet() {
    // [16, 56, 56, 64],[3, 3, 64, 64],[16, 64, 56, 56]
    test_conv(16*64*56*56, 64*64*3*3, 16*64*56*56, "../data/resnet/data2.bin", "../data/resnet/filter2.bin", 2, 4, 3136);
    // [16, 56, 56, 64],[3, 3, 64, 64],[16, 64, 56, 56],
    test_conv(16*64*56*56, 64*64*3*3, 16*64*56*56, "../data/resnet/data3.bin", "../data/resnet/filter3.bin", 3, 4, 3136);
    // [16, 56, 56, 64],[3, 3, 64, 64],[16, 64, 56, 56],
    test_conv(16*64*56*56, 64*64*3*3, 16*64*56*56, "../data/resnet/data4.bin", "../data/resnet/filter4.bin", 4, 4, 3136);
    // [16, 56, 56, 64],[3, 3, 64, 64],[16, 64, 56, 56],
    test_conv(16*64*56*56, 64*64*3*3, 16*64*56*56, "../data/resnet/data5.bin", "../data/resnet/filter5.bin", 5, 4, 3136);
    // [16, 56, 56, 64],[3, 3, 64, 128],[16, 128, 28, 28],
    test_conv(16*64*56*56, 64*128*3*3, 16*128*28*28, "../data/resnet/data6.bin", "../data/resnet/filter6.bin", 6, 8, 784);
    // [16, 28, 28, 128],[3, 3, 128, 128],[16, 128, 28, 28],
    test_conv(16*28*28*128, 128*128*3*3, 16*128*28*28, "../data/resnet/data7.bin", "../data/resnet/filter7.bin", 7, 8, 784);
    // [16, 28, 28, 128],[3, 3, 128, 128],[16, 128, 28, 28],
    test_conv(16*28*28*128, 128*128*3*3, 16*128*28*28, "../data/resnet/data8.bin", "../data/resnet/filter8.bin", 8, 8, 784);
    // [16, 28, 28, 128],[3, 3, 128, 128],[16, 128, 28, 28],
    test_conv(16*28*28*128, 128*128*3*3, 16*128*28*28, "../data/resnet/data9.bin", "../data/resnet/filter9.bin", 9, 8, 784);
    // [16, 28, 28, 128],[3, 3, 128, 256],[16, 256, 14, 14],
    test_conv(16*28*28*128, 128*256*3*3, 16*256*14*14, "../data/resnet/data10.bin", "../data/resnet/filter10.bin", 10, 16, 196);
    // [16, 14, 14, 256],[3, 3, 256, 256],[16, 256, 14, 14],
    test_conv(16*14*14*256, 256*256*3*3, 16*256*14*14, "../data/resnet/data11.bin", "../data/resnet/filter11.bin", 11, 16, 196);
    // [16, 14, 14, 256],[3, 3, 256, 256],[16, 256, 14, 14],
    test_conv(16*14*14*256, 256*256*3*3, 16*256*14*14, "../data/resnet/data12.bin", "../data/resnet/filter12.bin", 12, 16, 196);
    // [16, 14, 14, 256],[3, 3, 256, 256],[16, 256, 14, 14],
    test_conv(16*14*14*256, 256*256*3*3, 16*256*14*14, "../data/resnet/data13.bin", "../data/resnet/filter13.bin", 13, 16, 196);
    // [16, 14, 14, 256],[3, 3, 256, 512],[16, 512, 7, 7],
    test_conv(16*14*14*256, 256*512*3*3, 16*512*7*7, "../data/resnet/data14.bin", "../data/resnet/filter14.bin", 14, 32, 49);
    // [16, 7, 7, 512],[3, 3, 512, 512],[16, 512, 7, 7],
    test_conv(16*7*7*512, 512*512*3*3, 16*512*7*7, "../data/resnet/data15.bin", "../data/resnet/filter15.bin", 15, 32, 49);
    // [16, 7, 7, 512],[3, 3, 512, 512],[16, 512, 7, 7],
    test_conv(16*7*7*512, 512*512*3*3, 16*512*7*7, "../data/resnet/data16.bin", "../data/resnet/filter16.bin", 16, 32, 49);
    // [16, 7, 7, 512],[3, 3, 512, 512],[16, 512, 7, 7]
    test_conv(16*7*7*512, 512*512*3*3, 16*512*7*7, "../data/resnet/data17.bin", "../data/resnet/filter17.bin", 17, 32, 49);

}

int main() {
    test_resnet();
}