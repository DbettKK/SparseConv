#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <string>

static const int M = 16;
static const int N = 8;
static const int K = 16;

__global__ void sparse_mma(half *d, half *a, half *b, half *c, const uint32_t *metadata_p) {
    uint32_t tid = threadIdx.x;
    uint32_t metadata = metadata_p[tid / 4];
    __half *a_ptr = a + (tid % 4) * 2 + (tid / 4) * 8;
    __half *b_ptr = b + (tid % 4) * 2 * 8 + tid / 4;
    __half *c_ptr = c + (tid % 4) * 2 + (tid / 4) * 8;
    __half *d_ptr = d + (tid % 4) * 2 + (tid / 4) * 8;
    asm volatile("{\n\t"
             ".reg .f16 %Ra_single<4>, %Rb_single<4>;\n\t"
             ".reg .f16x2 %Ra<2>, %Rb<2>, %Rc<2>, %Rd<2>;\n\t"
             "ld.global.ca.b32 %Ra0, [%1];\n\t"
             "ld.global.ca.b32 %Ra1, [%1 + 128];\n\t"
             "ld.global.ca.b16 %Rb_single0, [%2];\n\t"
             "ld.global.ca.b16 %Rb_single1, [%2 + 16];\n\t"
             "ld.global.ca.b16 %Rb_single2, [%2 + 128];\n\t"
             "ld.global.ca.b16 %Rb_single3, [%2 + 144];\n\t"
             "ld.global.ca.b32 %Rc0, [%3];\n\t"
             "ld.global.ca.b32 %Rc1, [%3 + 128];\n\t"
             "mov.b32 %Rb0, {%Rb_single0, %Rb_single1};\n\t"
             "mov.b32 %Rb1, {%Rb_single2, %Rb_single3};\n\t"
             "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16\n\t"
             "{%Rd0, %Rd1},\n\t"
             "{%Ra0, %Ra1},\n\t"
             "{%Rb0, %Rb1},\n\t"
             "{%Rc0, %Rc1}, %4, 0x0;\n\t"
             "st.global.wb.b32 [%0], %Rd0;\n\t"
             "st.global.wb.b32 [%0 + 128], %Rd1;\n\t"
             "}\n\t"
        :
        : "l"(d_ptr), "l"(a_ptr), "l"(b_ptr), "l"(c_ptr), "r"(metadata)
        : "memory");
}


// 默认matB为16*16
__global__ void test_mma(half *matA_cmpr, half* matB, half* zeroC, half* calA, half* calD, uint32_t* meta, int m, int k, half *out) {
    int idx = threadIdx.x, idy = threadIdx.y;
    int row_start = idx * 16, col_start = idy * 8;
    int index = idx * (k / 8) + idy;
    // problem
    half *calA_this = calA + index * 128;
    half *calD_this = calD + index * 128;
    int cnt = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            calA_this[cnt++] = matA_cmpr[(i + row_start) * k + (j + col_start)];
        }
    }
    sparse_mma<<<1, 32>>>(calD_this, calA_this, matB, zeroC, meta + 8 * index);
    printf("\n"); // 玄学
    cnt = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            out[(i + row_start) * k + (j + col_start)] = calD_this[cnt++];
        }
    }
}

int main1(int argc, char **argv) {
//    size_t mat_a_size = M * K / 2;
//    size_t mat_b_size = N * K;
//    size_t mat_c_d_size = M * N;
//    size_t metadata_size_bytes = M * 2; // 16 bit per row
    half *hA_cmpr = new half[M * K / 2];
    half *hB = new half[K * N];
    half *hC = new half[M * N];
    half *hD = new half[M * N];
    auto *hMeta = new uint32_t[8];
    // uint32_t meta = 0b10001000100010001000100010001000; // 2290649224
    uint32_t meta = 2290649224; // [x,0,x,0] -> index:10,00
    for (int i = 0; i < 8; i++) hMeta[i] = meta;
    for (int i = 0; i < M * K / 2; i++) hA_cmpr[i] = __float2half(1.0);
    for (int i = 0; i < K * N; i++) hB[i] = __float2half(1.0);
    for (int i = 0; i < M * N; i++) hC[i] = __float2half(0.0);
    for (int i = 0; i < M * N; i++) hD[i] = __float2half(0.0);

    half *dA_cmpr, *dB, *dC, *dD;
    uint32_t *dMeta;
    cudaMalloc((void **)&dA_cmpr, sizeof(half) * M * K / 2);
    cudaMalloc((void **)&dB, sizeof(half) * N * K);
    cudaMalloc((void **)&dC, sizeof(half) * M * N);
    cudaMalloc((void **)&dD, sizeof(half) * M * N);
    cudaMalloc((void **)&dMeta, M * 2);

    cudaMemcpy(dA_cmpr, hA_cmpr, sizeof(half) * M * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(half) * N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeof(half) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dD, hD, sizeof(half) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dMeta, hMeta, M * 2, cudaMemcpyHostToDevice);

    sparse_mma<<<1, 32>>>(dD, dA_cmpr, dB, dC, dMeta);
    // cudaDeviceSynchronize();

    cudaMemcpy(hD, dD, sizeof(half) * M * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", __half2int_rz(hD[i * N + j]));
        }
        printf("\n");
    }
    return 0;
}

int main7() {
    half *hA = new half[32 * 16];
    half *hB = new half[16 * 8];
    auto *hMeta = new uint32_t[8 * 4];
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            int index = i / 16 * 2 + j / 8;
            hA[i * 16 + j] = index + 1;
        }
    }
    for (int i = 0; i < 16 * 8; i++) hB[i] = 1;
    for (int i = 0; i < 32; i++) hMeta[i] = 0b10001000100010001000100010001000;

    half *dA, *dB, *dC, *dCalA, *dCalD, *dOut;
    cudaMalloc((void **)&dA, sizeof(half) * 32 * 16);
    cudaMalloc((void **)&dB, sizeof(half) * 16 * 8);
    cudaMalloc((void **)&dC, sizeof(half) * 16 * 16);
    cudaMalloc((void **)&dCalA, sizeof(half) * 32 * 16);
    cudaMalloc((void **)&dCalD, sizeof(half) * 32 * 16);
    cudaMalloc((void **)&dOut, sizeof(half) * 32 * 16);
    cudaMemcpy(dA, hA, sizeof(half) * 32 * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(half) * 16 * 8, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, sizeof(half) * 16 * 16);
    cudaMemset(dCalA, 0, sizeof(half) * 32 * 16);
    cudaMemset(dCalD, 0, sizeof(half) * 32 * 16);
    cudaMemset(dOut, 0, sizeof(half) * 32 * 16);

    uint32_t* dMeta;
    cudaMalloc((void **)&dMeta, sizeof(uint32_t) * 8 * 4);
    cudaMemcpy(dMeta, hMeta, sizeof(uint32_t) * 8 * 4, cudaMemcpyHostToDevice);

    dim3 blocks(2,2,1);
    test_mma<<<1, blocks>>>(dA, dB, dC, dCalA, dCalD, dMeta, 32, 16, dOut);

    half *hOut = new half[32 * 16];
    cudaMemcpy(hOut, dOut, sizeof(half) * 32 * 16, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%d ", __half2int_rz(hOut[i * 16 + j]));
        }
        printf("\n");
    }
    return 0;
}