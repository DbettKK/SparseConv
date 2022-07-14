//
// Created by dbettkk on 2022/7/14.
//

#include "interface/sparse_mma_gemm.cuh"

int main() {
    half *hA = new half[256];
    half *hB = new half[256];
    half *hOut = new half[256];

    for (int i = 0; i < 256; i+=2) {
        hA[i] = 2;
        hA[i + 1] = 0;
    }
    for (int i = 0; i < 256; i++) hB[i] = 1;

    sparse_mma_gemm_device(hA, hB, 16, 16, 16, true, hOut);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%d ", __half2int_rz(hOut[i]));
        }
        printf("\n");
    }
    return 0;
}