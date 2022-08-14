//
// Created by dbettkk on 2022/8/14.
//

#include "test_spmma.cuh"

float rand_float() {
    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_real_distribution<float> u(0, 5); // 闭区间
    return u(e);
}

void test_pad_time() {
    int batch = 1, m = 64, k = 64, n = 64;
    auto test = new Test(batch, m, k, n);

    half *dA, *dB, *dOut, *dOut2;
    test->generate_sparse_A(dA);
    test->generate_dense_B(dB);
    test->generate_zero_C(dOut);
    test->generate_zero_C(dOut2);

    sparse_mma_gemm_noPad_device(dA, dB, m, k, n, true, dOut);
    sparse_mma_gemm_device(dA, dB, m, k, n, true, dOut2);

    test->matC_diff(dOut, dOut2);
}
