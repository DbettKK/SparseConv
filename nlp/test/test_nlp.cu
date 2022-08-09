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

void test_spmma_cublas() {
    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_real_distribution<float> u(0, 10); // 闭区间

    int m = 4, k = 16, n = 64;
    half *hA = new half[m * k];
    half *hB = new half[k * n];
    for (int i = 0; i < m * k; i+=2) {
        hA[i] = 0;
        hA[i + 1] = 4;
    }
    for (int i = 0; i < k * n; i++) hB[i] = 5;

    half *dA, *dB, *d1, *d2;
    cudaMalloc(&dA, sizeof(half) * m * k);
    cudaMalloc(&dB, sizeof(half) * n * k);
    cudaMalloc(&d1, sizeof(half) * m * n);
    cudaMalloc(&d2, sizeof(half) * m * n);
    cudaMemcpy(dA, hA, sizeof(half) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(half) * n * k, cudaMemcpyHostToDevice);
    cublas_gemm_device(dA, dB, m, k, n, d1);
    sparse_mma_gemm_splitK_device(dA, dB, m, k, n, false, d2);

    half *o1 = new half[m * n];
    half *o2 = new half[m * n];
    cudaMemcpy(o1, d1, sizeof(half) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(o2, d2, sizeof(half) * m * n, cudaMemcpyDeviceToHost);

    half *hC = new half[m * n];
    memset(hC, 0, sizeof(half) * m * n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int v = 0; v < k; v++) {
                hC[i * n + j] = __half2float(hC[i * n + j]) + __half2float(hA[i * k + v]) * __half2float(hB[v * n + j]);
            }
        }
    }

    int diff = 0;
    for (int i = 0; i < m * n; i++) {
        if (__half2float(o1[i]) != __half2float(o2[i])) diff++;
        //printf("diff: %.2f : %.2f\n", __half2float(o1[i]), __half2float(o2[i]));
    }
    printf("cublas, spmma: total: %d, diff: %d\n", m * n, diff);
    diff = 0;
    for (int i = 0; i < m * n; i++) {
        if (__half2float(hC[i]) != __half2float(o2[i])) diff++;
        //printf("diff: %.2f : %.2f\n", __half2float(o1[i]), __half2float(o2[i]));
    }
    printf("spmma, cpu: total: %d, diff: %d\n", m * n, diff);
    diff = 0;
    for (int i = 0; i < m * n; i++) {
        if (__half2float(o1[i]) != __half2float(hC[i])) diff++;
        //printf("diff: %.2f : %.2f\n", __half2float(o1[i]), __half2float(hC[i]));
    }
    printf("cublas, cpu: total: %d, diff: %d\n", m * n, diff);
    printf("cublas:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", __half2float(o1[i]));
        }
        printf("\n");
    }
    printf("cpu:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", __half2float(hC[i]));
        }
        printf("\n");
    }
    printf("spmma:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", __half2float(o2[i]));
        }
        printf("\n");
    }
}