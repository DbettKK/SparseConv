//
// Created by dbettkk on 2022/8/14.
//

#include "test_spmma.cuh"

void test_pad_time() {
    int batch = 2, m = 512, k = 512, n = 64;
    auto test = new Test(batch, m, k, n);

    half *dA, *dB, *dOut, *dOut2;
    test->generate_sparse_A(&dA);
    test->generate_dense_B(&dB);
    test->generate_zero_C(&dOut);
    test->generate_zero_C(&dOut2);

    for (int i = 0; i < 5; i++) {
        auto t = new CudaTime();
        t->initAndStart();
        for (int j = 0; j < batch; j++) {
            sparse_mma_gemm_noPad_device(dA + j * m * k, dB + j * k * n, m, k, n, true, dOut + j * m * n);
        }
        printf("spmma no pad time: %fms\n", t->endAndGetTime());
    }
    for (int i = 0; i < 5; i++) {
        auto t = new CudaTime();
        t->initAndStart();
        for (int j = 0; j < batch; j++) {
            sparse_mma_gemm_device(dA + j * m * k, dB + j * k * n, m, k, n, true, dOut2 + j * m * n);
        }
        printf("spmma pad time: %fms\n", t->endAndGetTime());
    }
    for (int i = 0; i < 5; i++) {
        auto t = new CudaTime();
        t->initAndStart();
        for (int j = 0; j < batch; j++) {
            cublas_gemm_device(dA + j * m * k, dB + j * k * n, m, k, n, dOut2 + j * m * n);
        }
        printf("cublas time: %fms\n", t->endAndGetTime());
    }
    for (int i = 0; i < 5; i++) {
        auto t = new CudaTime();
        t->initAndStart();
        sparse_mma_gemm_noPad_batches_device(dA, dB, batch, m, k, n, true, dOut);
        printf("spmma batch no pad time: %fms\n", t->endAndGetTime());
    }
    for (int i = 0; i < 5; i++) {
        auto t = new CudaTime();
        t->initAndStart();
        sparse_mma_gemm_batches_device(dA, dB, batch, m, k, n, true, dOut2);
        printf("spmma batch pad time: %fms\n", t->endAndGetTime());
    }
    for (int i = 0; i < 5; i++) {
        auto t = new CudaTime();
        t->initAndStart();
        cublas_gemm_batches_device(dA, dB, batch, m, k, n, false, dOut2);
        printf("cublas batch time: %fms\n", t->endAndGetTime());
    }
    for (int i = 0; i < 5; i++) {
        auto t = new CudaTime();
        t->initAndStart();
        cublas_gemm_batches_device_v2(dA, dB, batch, m, k, n, false, dOut2);
        printf("cublas batch v2 time: %fms\n", t->endAndGetTime());
    }
    test->matC_diff(dOut, dOut2);

}

void test_cusparse() {
    int batch = 2, m = 512, k = 512, n = 64;
    auto test = new Test(batch, m, k, n);

    half *dA, *dB, *dOut, *dOut2;
    test->generate_sparse_A(&dA);
    test->generate_dense_B(&dB);
    test->generate_zero_C(&dOut);
    test->generate_zero_C(&dOut2);

    sparse_mma_gemm_batches_device(dA, dB, 2, m, k, n, true, dOut);
    cusparse_gemm_coo_batched_device(dA, dB, 2, m, k, n, dOut2);
    test->matC_diff(dOut, dOut2);
}
