//
// Created by dbettkk on 2022/6/24.
//
#include<iostream>
#include<ctime>
#include <cstdlib>

void spmma_cpu(const int *matA_cmpr, const int *matB, const int *meta, int *out) {
    // 16x16x16
    memset(out, 0, sizeof(int) * 128);
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                out[i * 8 + k] += matA_cmpr[i * 8 + j] * matB[meta[i * 8 + j] * 8 + k];
            }
        }
    }
}

void normal_cpu(const int *matA, const int *matB, int *out) {
    memset(out, 0, sizeof(int) * 128);
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            for (int k = 0; k < 8; k++) {
                out[i * 8 + k] += matA[i * 16 + j] * matB[j * 8 + k];
            }
        }
    }
}

int main() {
    int *hA_big = new int[16 * 16];
    int *hA = new int[16*8];
    int *hB = new int[16*8];
    int *out = new int[16*8];
    int *meta = new int[16*8];

    for (int i = 0; i < 256; i++) hA_big[i] = 1;
    for (int i = 0; i < 128; i++) hA[i] = 1;
    for (int i = 0; i < 128; i++) hB[i] = 1;
    for (int i = 0; i < 128; i++) meta[i] = rand() % 16;
    for (int i = 0; i < 128; i++) out[i] = 0;

    clock_t start, end;
    start = clock();
    for (int i = 0; i < 100000; i++) spmma_cpu(hA, hB, meta, out);
    //for (int i = 0; i < 100000; i++) normal_cpu(hA_big, hB, out);
    end = clock();
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%d ", out[i * 8 + j]);
        }
        printf("\n");
    }

    printf("time: %ld clocks\n", end - start);
}
