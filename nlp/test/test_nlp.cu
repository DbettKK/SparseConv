//
// Created by dbettkk on 2022/8/8.
//

#include "test_nlp.cuh"

void test_gemm_batches() {
    const int batch = 4, row = 16, col = 16;
    int size = batch * row * col;
    half *hA = new half[size];
    half *hB = new half[size];
    for (int i = 0; i < size; i++) {
        hA[i] = 4;
        hB[i] = 5;
        //hA[i] = 2;
        //hB[i] = 3;
    }
    half *hOut = new half[size];
    half *hOut2 = new half[size];
    half *dA, *dB, *dOut, *dOut2;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(half) * size))
    CHECK_CUDA(cudaMalloc(&dB, sizeof(half) * size))
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(half) * size))
    CHECK_CUDA(cudaMalloc(&dOut2, sizeof(half) * size))
    CHECK_CUDA(cudaMemcpy(dA, hA, sizeof(half) * size, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB, hB, sizeof(half) * size, cudaMemcpyHostToDevice))

    for (int i = 0; i < batch; i++) {
        int each = i * row * col;
        cublas_gemm_device(dA + each, dB + each, 16, 16, 16, dOut + each);
    }

    cublas_gemm_batches_device(dA, dB, batch, 16, 16, 16, false, dOut2);

    CHECK_CUDA(cudaMemcpy(hOut, dOut, sizeof(half) * size, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(hOut2, dOut2, sizeof(half) * size, cudaMemcpyDeviceToHost))
    int diff = 0;
    for (int i = 0; i < size; i++) {
        if (__half2float(hOut[i]) != __half2float(hOut2[i])) {
            diff++;
            printf("diff: %f : %f\n", __half2float(hOut[i]), __half2float(hOut2[i]));
        }
    }
    printf("total: %d, diff: %d\n", size, diff);
}