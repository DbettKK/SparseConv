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
    for (int i = 0; i < m * k; i += 2) {
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
    sparse_mma_gemm_device(dA, dB, m, k, n, false, d2);

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

void test_spmma_cublas_efficient() {
    const int M = 64;
    const int K = 64;
    const int N = 64;
    int A_size = M * K;
    int B_size = N * K;
    int C_size = M * N;

    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_real_distribution<float> u(0, 5); // 闭区间

    half *hA = new half[A_size];
    half *hB = new half[B_size];
    for (int i = 0; i < A_size; i += 2) {
        hA[i] = u(e);
        hA[i + 1] = 0;
    }
    for (int i = 0; i < B_size; i++) hB[i] = u(e);

    half *dA, *dB, *dOut1, *dOut2;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(half) * A_size))
    CHECK_CUDA(cudaMalloc(&dB, sizeof(half) * B_size))
    CHECK_CUDA(cudaMalloc(&dOut1, sizeof(half) * C_size))
    CHECK_CUDA(cudaMalloc(&dOut2, sizeof(half) * C_size))
    for (int i = 0; i < 10; i++) {
        auto t1 = new CudaTime();
        t1->initAndStart();
        sparse_mma_gemm_device(dA, dB, M, K, N, true, dOut1);
        printf("spmma time: %fms\n", t1->endAndGetTime());
    }
    for (int i = 0; i < 10; i++) {
        auto t2 = new CudaTime();
        t2->initAndStart();
        cublas_gemm_device(dA, dB, M, K, N, dOut1);
        printf("cublas time: %fms\n", t2->endAndGetTime());
    }

}

void test_spmma_batches() {
    const int batch = 64, row = 16, col = 16;
    int size = batch * row * col;
    half *hA = new half[size];
    half *hB = new half[size];
    for (int i = 0; i < size; i += 2) {
        hA[i] = 4;
        hA[i + 1] = 0;
        hB[i] = 5;
        hB[i + 1] = 5;
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
    for (int j = 0; j < 4; j++) {
        auto tt = new CudaTime();
        tt->initAndStart();
        for (int i = 0; i < batch; i++) {
            int each = i * row * col;
            sparse_mma_gemm_device(dA + each, dB + each, row, row, row, true, dOut + each);
        }
        printf("spmma time: %fms\n", tt->endAndGetTime());
    }
    for (int j = 0; j < 4; j++) {
        auto tt1 = new CudaTime();
        tt1->initAndStart();
        for (int i = 0; i < batch; i++) {
            int each = i * row * col;
            cublas_gemm_device(dA + each, dB + each, row, row, row, dOut + each);
        }
        printf("cublas time: %fms\n", tt1->endAndGetTime());
    }
    for (int i = 0; i < 4; i++) {
        auto tt2 = new CudaTime();
        tt2->initAndStart();
        sparse_mma_gemm_batches_device(dA, dB, batch, row, row, row, true, dOut);
        printf("spmma batch time: %fms\n", tt2->endAndGetTime());
    }
    for (int i = 0; i < 4; i++) {
        auto tt3 = new CudaTime();
        tt3->initAndStart();
        cublas_gemm_batches_device(dA, dB, batch, row, row, row, false, dOut2);
        printf("cublas batch time: %fms\n", tt3->endAndGetTime());
    }
    CHECK_CUDA(cudaMemcpy(hOut, dOut, sizeof(half) * size, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(hOut2, dOut2, sizeof(half) * size, cudaMemcpyDeviceToHost))
    int diff = 0;
    for (int i = 0; i < size; i++) {
        if (__half2float(hOut[i]) != __half2float(hOut2[i])) {
            diff++;
            printf("diff: %f : %f\n", __half2float(hOut[i]), __half2float(hOut2[i]));
        }
    }
    printf("e.g.: %.2f, %.2f\n", __half2float(hOut[1]), __half2float(hOut2[1]));
    printf("total: %d, diff: %d\n", size, diff);
}

void test_transpose_batches() {
    int batch = 64, m = 16, n = 64;
    int size = batch * m * n;
    half *hA = new half[size];
    for (int i = 0; i < size; i++) hA[i] = rand() % 10;

    half *dA, *dOut, *dOut2;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(half) * size));
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(half) * size));
    CHECK_CUDA(cudaMalloc(&dOut2, sizeof(half) * size));
    CHECK_CUDA(cudaMemcpy(dA, hA, sizeof(half) * size, cudaMemcpyHostToDevice))

    dim3 grid(batch / 32 + 1, m, n);
    for (int i = 0; i < 5; i++) {
        auto tt = new CudaTime();
        tt->initAndStart();
        transpose_batches<<<grid, 32>>>(dA, dOut2, batch, m, n);
        printf("transpose batch time: %fms\n", tt->endAndGetTime());
    }

    for (int i = 0; i < 5; i++) {
        auto tt = new CudaTime();
        tt->initAndStart();
        for (int j = 0; j < batch; j++) {
            dim3 g(m / 32 + 1, n / 32 + 1);
            dim3 t(32, 32);
            transpose_half<<<g, t>>>(dA + j * m * n, dOut + j * m * n, m, n);
        }
        printf("transpose time: %fms\n", tt->endAndGetTime());
    }

    half *out = new half[size];
    half *out2 = new half[size];
    CHECK_CUDA(cudaMemcpy(out, dOut, sizeof(half) * size, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(out2, dOut2, sizeof(half) * size, cudaMemcpyDeviceToHost))

    int diff = 0;
    for (int i = 0; i < size; i++) {
        if (__half2float(out[i]) != __half2float(out2[i])) diff++;
    }
    printf("total: %d, diff: %d", size, diff);
}

void test_softmax() {
    int batch = 64, row = 16, col = 64;
    int size = batch * row * col;
    half *hA = new half[size];
    for (int i = 0; i < size; i++) hA[i] = i;

    half *dA, *dOut, *dOut2;
    cudaMalloc(&dA, sizeof(half) * size);
    cudaMalloc(&dOut, sizeof(half) * size);
    cudaMalloc(&dOut2, sizeof(half) * size);
    cudaMemcpy(dA, hA, sizeof(half) * size, cudaMemcpyHostToDevice);


    half *hOut = new half[size];
    half *hOut2 = new half[size];
    for (int i = 0; i < 5; i++) {
        auto tt0 = new CudaTime();
        tt0->initAndStart();
        dim3 grid(row, batch);
        softmax_batches<<<grid, col>>>(dA, batch, row, col, dOut);
        printf("v1 time: %fms\n", tt0->endAndGetTime());
    }
    for (int i = 0; i < 5; i++) {
        auto tt1 = new CudaTime();
        tt1->initAndStart();
        for (int j = 0; j < batch; j++) {
            softmax_half_v2<<<row, col>>>(dA + i * col * row, col, col, dOut2 + i * col * row);

        }
        printf("v2 time: %fms\n", tt1->endAndGetTime());
    }


    cudaMemcpy(hOut, dOut, sizeof(half) * 1024, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOut2, dOut2, sizeof(half) * 1024, cudaMemcpyDeviceToHost);

    int diff = 0;
    for (int i = 0; i < size; i++) {
        if (__half2float(hOut[i]) != __half2float(hOut2[i])) diff++;
    }
    printf("total: %d, diff: %d", size, diff);
//    for (int i = 0; i < 16; i++) {
//        for (int j = 0; j < 64; j++) {
//            printf("%.2f ", __half2float(hOut[i * 64 + j]));
//        }
//        printf("\n");
//    }
//    for (int i = 0; i < 16; i++) {
//        for (int j = 0; j < 64; j++) {
//            printf("%.2f ", __half2float(hOut2[i * 64 + j]));
//        }
//        printf("\n");
//    }
}

__global__ void share(half *item, int size, half *sum) {
    // 优化：先把所有元素放入share memory
    __shared__ float mem;
    int tid = threadIdx.x;
    if (tid == 1) {
        float s = 0;
        for (int i = 0; i < size; i++) {
            s += __half2float(item[i]);
        }
        mem = s;
    }
    __syncthreads();

    sum[tid] = mem;
}

void test_shared_mem() {
    half *item = new half[32];
    for (int i = 0; i < 32; i++) item[i] = 2;
    half *dA, *dOut;
    cudaMalloc(&dA, sizeof(half) * 32);
    cudaMalloc(&dOut, sizeof(half) * 32);
    cudaMemcpy(dA, item, sizeof(half) * 32, cudaMemcpyHostToDevice);

    share<<<1, 32>>>(dA, 32, dOut);
    half *out = new half[32];
    cudaMemcpy(out, dOut, sizeof(half) * 32, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32; i++) printf("%d ", __half2int_rz(out[i]));
}