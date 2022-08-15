//
// Created by dbettkk on 2022/8/14.
//

#include "Test.cuh"

Test::Test(int batch, int m, int k, int n) : batch(batch), m(m), k(k), n(n) {}

int Test::matA_size() const {
    return batch * m * k;
}

int Test::matB_size() const {
    return batch * n * k;
}

int Test::matC_size() const {
    return batch * m * n;
}

float Test::randf() {
    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_real_distribution<float> u(0, 5); // 闭区间
    return u(e);
}

float Test::randf(int bound) {
    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_real_distribution<float> u(0, bound); // 闭区间
    return u(e);
}

void Test::generate_sparse_A(half **matA) {
    half *hA = new half[matA_size()];
    for (int i = 0; i < matA_size(); i+=2) {
        hA[i] = randf();
        hA[i + 1] = 0;
    }
    CHECK_CUDA(cudaMalloc(matA, sizeof(half) * matA_size()))
    CHECK_CUDA(cudaMemcpy(*matA, hA, sizeof(half) * matA_size(), cudaMemcpyHostToDevice))

    delete[] hA;
}

void Test::generate_dense_B(half **matB) {
    half *hB = new half[matB_size()];
    for (int i = 0; i < matB_size(); i++) hB[i] = randf();
    CHECK_CUDA(cudaMalloc(matB, sizeof(half) * matB_size()))
    CHECK_CUDA(cudaMemcpy(*matB, hB, sizeof(half) * matB_size(), cudaMemcpyHostToDevice))
    delete[] hB;
}

void Test::generate_zero_C(half **matC) const {
    CHECK_CUDA(cudaMalloc(matC, sizeof(half) * matC_size()))
    CHECK_CUDA(cudaMemset(*matC, 0, sizeof(half) * matC_size()))
}

void Test::matC_diff(half *matC1, half *matC2) const {
    half *hC1 = new half[matC_size()];
    half *hC2 = new half[matC_size()];
    CHECK_CUDA(cudaMemcpy(hC1, matC1, sizeof(half) * matC_size(), cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(hC2, matC2, sizeof(half) * matC_size(), cudaMemcpyDeviceToHost))

    int diff = 0;
    for (int i = 0; i < matC_size(); i++) {
        if (__half2float(hC1[i]) != __half2float(hC2[i])) {
            diff++;
        }
    }
    printf("total: %d, diff: %d\n", matC_size(), diff);

    delete[] hC1;
    delete[] hC2;
}

void Test::print_matC(half *matC) const {
    half *hC = new half[matC_size()];
    CHECK_CUDA(cudaMemcpy(hC, matC, sizeof(half) * matC_size(), cudaMemcpyDeviceToHost))

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < m; j++) {
            for (int v = 0; v < n; v++) {
                printf("%.2f ", __half2float(hC[i * m * n + j * n + v]));
            }
            printf("\n");
        }
        printf("\n");
    }
    delete[] hC;
}



