//
// Created by dbettkk on 2022/8/14.
//
#include "cublas_interface.cuh"

void cublas_gemm_device(const half *d_A, const half *d_B, int inputM, int inputK, int inputN, half *output) {
    // 因为为列存储，为了方便，设置转置
    cublasHandle_t cublasH = nullptr;

    const int m = inputM;
    const int n = inputN;
    const int k = inputK;
    const int lda = k; // 因为转置了 因此ld代表列数
    const int ldb = n;
    const int ldc = m; // c的ld都是m

    const half alpha = 1.0;
    const half beta = 0.0;

    half *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_T;

    /* step 1: create cublas handle, bind a stream */
    CHECK_CUBLAS( cublasCreate(&cublasH) );

    /* step 2: copy data to device */
    CHECK_CUDA( cudaMalloc(&d_C, sizeof(half) * m * n) );

    /* step 3: compute */
    CHECK_CUBLAS( cublasHgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc) );

    // transpose
    dim3 grid(m / 32 + 1, n / 32 + 1);
    dim3 block(32, 32);
    transpose_half<<<grid, block>>>(d_C, output, m, n);

    /* step 4: copy data to host */
    //CHECK_CUDA( cudaMemcpyAsync(output, d_C, sizeof(half) * m * n, cudaMemcpyDeviceToDevice, stream));

    /* free resources */
    CHECK_CUDA( cudaFree(d_C) );
    CHECK_CUBLAS( cublasDestroy(cublasH) );

}

void cublas_gemm_device_scale(const half *d_A, const half *d_B, int inputM, int inputK, int inputN, float scale, half *output) {
    // 因为为列存储，为了方便，设置转置
    cublasHandle_t cublasH = nullptr;

    const int m = inputM;
    const int n = inputN;
    const int k = inputK;
    const int lda = k; // 因为转置了 因此ld代表列数
    const int ldb = n;
    const int ldc = m; // c的ld都是m

    const half alpha = 1.0;
    const half beta = 0.0;

    half *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_T;

    /* step 1: create cublas handle, bind a stream */
    CHECK_CUBLAS( cublasCreate(&cublasH) );

    /* step 2: copy data to device */
    CHECK_CUDA( cudaMalloc(&d_C, sizeof(half) * m * n) );

    /* step 3: compute */
    CHECK_CUBLAS( cublasHgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc) );

    CHECK_CUBLAS( cublasScalEx(cublasH, m * n, &scale, CUDA_R_32F, d_C, CUDA_R_16F, 1, CUDA_R_32F) )

    // transpose
    dim3 grid(m / 32 + 1, n / 32 + 1);
    dim3 block(32, 32);
    transpose_half<<<grid, block>>>(d_C, output, m, n);

    /* step 4: copy data to host */
    //CHECK_CUDA( cudaMemcpyAsync(output, d_C, sizeof(half) * m * n, cudaMemcpyDeviceToDevice, stream));

    /* free resources */
    CHECK_CUDA( cudaFree(d_C) );
    CHECK_CUBLAS( cublasDestroy(cublasH) );

}

void cublas_gemm_batches_device(half *d_A, half *d_B, int batch, int inputM, int inputK, int inputN,
                                bool isSingleBatch, half *output) {
    const int m = inputM;
    const int n = inputN;
    const int k = inputK;
    const int lda = k; // 因为转置了 因此ld代表列数
    const int ldb = n;
    const int ldc = m; // c的ld都是m

    const half alpha = 1.0;
    const half beta = 0.0;

    half *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_T;

    /* step 1: create cublas handle, bind a stream */
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS( cublasCreate(&handle) );

    /* step 2: copy data to device */
    CHECK_CUDA( cudaMalloc(&d_C, sizeof(half) * batch * m * n) );
    CHECK_CUDA( cudaMemset(d_C, 0, sizeof(half) * batch * m * n) );

    /* step 3: compute */
    half **dArrA, **dArrB, **dArrC;
    half *arrA[64], *arrB[64], *arrC[64];
    if (isSingleBatch) {
        for (int i = 0; i < batch; i++) {
            half *tmpB;
            CHECK_CUDA(cudaMalloc(&tmpB, sizeof(half) * k * n))
            CHECK_CUDA(cudaMemcpy(tmpB, d_B, sizeof(half) * k * n, cudaMemcpyDeviceToDevice))
            arrA[i] = d_A + i * m * k;
            arrB[i] = tmpB;
            arrC[i] = d_C + i * m * n;
        }
    } else {
        for (int i = 0; i < batch; i++) {
            arrA[i] = d_A + i * m * k;
            arrB[i] = d_B + i * n * k;
            arrC[i] = d_C + i * m * n;
        }
    }
    CHECK_CUDA(cudaMalloc(&dArrA, sizeof(half*) * batch))
    CHECK_CUDA(cudaMalloc(&dArrB, sizeof(half*) * batch))
    CHECK_CUDA(cudaMalloc(&dArrC, sizeof(half*) * batch))
    CHECK_CUDA(cudaMemcpy(dArrA, arrA, sizeof(half*) * batch, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dArrB, arrB, sizeof(half*) * batch, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dArrC, arrC, sizeof(half*) * batch, cudaMemcpyHostToDevice))

    CHECK_CUBLAS(cublasHgemmBatched(handle, transa, transb, m, n, k, &alpha, dArrA, lda, dArrB, ldb, &beta, dArrC, ldc, batch))

    CHECK_CUDA(cudaMemcpy(arrC, dArrC, sizeof(half*) * batch, cudaMemcpyDeviceToHost))

    for (int i = 0; i < batch; i++) {
        CHECK_CUDA(cudaMemcpy(output + i * m * n, arrC[i], sizeof(half) * m * n, cudaMemcpyDeviceToDevice));
    }
//    for (int i = 0; i < batch; i++) {
//        half *c_out = new half[m * n];
//        cudaMemcpy(c_out, arrC[i], sizeof(half) * m * k, cudaMemcpyDeviceToHost);
//        //cudaMemcpy(c_out, dArrC + i, sizeof(half) * m * n, cudaMemcpyDeviceToHost);
//        for (int j = 0; j < m; j++) {
//            for (int v = 0; v < n; v++) {
//                printf("%d ", __half2int_rz(c_out[j * n + v]));
//            }
//            printf("\n");
//        }
//    }
    /* free resources */
    CHECK_CUDA( cudaFree(d_C) );
    CHECK_CUDA( cudaFree(dArrA) );
    CHECK_CUDA( cudaFree(dArrB) );
    CHECK_CUDA( cudaFree(dArrC) );
    CHECK_CUBLAS( cublasDestroy(handle) );

}

void cublas_gemm_batches_device_v2(half *d_A, half *d_B, int batch, int inputM, int inputK, int inputN, bool isSingleBatch, half *output) {
    // 因为为列存储，为了方便，设置转置
    cublasHandle_t cublasH = nullptr;

    const int m = inputM;
    const int n = inputN;
    const int k = inputK;
    const int lda = k; // 因为转置了 因此ld代表列数
    const int ldb = n;
    const int ldc = m; // c的ld都是m

    const half alpha = 1.0;
    const half beta = 0.0;
    int64_t strideA = m * k, strideB = isSingleBatch ? 0 : k * n, strideC = m * n;

    half *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_T;

    /* step 1: create cublas handle, bind a stream */
    CHECK_CUBLAS( cublasCreate(&cublasH) );

    /* step 2: copy data to device */
    CHECK_CUDA( cudaMalloc(&d_C, sizeof(half) * batch * m * n) );

    /* step 3: compute */
    CHECK_CUBLAS( cublasHgemmStridedBatched(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, strideA,
                                            d_B, ldb, strideB, &beta, d_C, ldc, strideC, batch) );

    // transpose
    dim3 grid(batch / 32 + 1, m, n);
    transpose_batches<<<grid, 32>>>(d_C, output, batch, m, n);

    /* step 4: copy data to host */
    //CHECK_CUDA( cudaMemcpyAsync(output, d_C, sizeof(half) * m * n, cudaMemcpyDeviceToDevice, stream));

    /* free resources */
    CHECK_CUDA( cudaFree(d_C) );
    CHECK_CUBLAS( cublasDestroy(cublasH) );
}

void cublas_gemm_batches_scale_device(half *d_A, half *d_B, int batch, int inputM, int inputK, int inputN,
                                      float scale, half *output) {
    const int m = inputM;
    const int n = inputN;
    const int k = inputK;
    const int lda = k; // 因为转置了 因此ld代表列数
    const int ldb = n;
    const int ldc = m; // c的ld都是m

    const half alpha = 1.0;
    const half beta = 0.0;

    half *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_T;

    /* step 1: create cublas handle, bind a stream */
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS( cublasCreate(&handle) );

    /* step 2: copy data to device */
    CHECK_CUDA( cudaMalloc(&d_C, sizeof(half) * batch * m * n) );
    CHECK_CUDA( cudaMemset(d_C, 0, sizeof(half) * batch * m * n) );

    /* step 3: compute */
    half **dArrA, **dArrB, **dArrC;
    half *arrA[64], *arrB[64], *arrC[64];
    for (int i = 0; i < batch; i++) {
        arrA[i] = d_A + i * m * k;
        arrB[i] = d_B + i * n * k;
        arrC[i] = d_C + i * m * n;
    }
    CHECK_CUDA(cudaMalloc(&dArrA, sizeof(half*) * batch))
    CHECK_CUDA(cudaMalloc(&dArrB, sizeof(half*) * batch))
    CHECK_CUDA(cudaMalloc(&dArrC, sizeof(half*) * batch))
    CHECK_CUDA(cudaMemcpy(dArrA, arrA, sizeof(half*) * batch, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dArrB, arrB, sizeof(half*) * batch, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dArrC, arrC, sizeof(half*) * batch, cudaMemcpyHostToDevice))

    CHECK_CUBLAS(cublasHgemmBatched(handle, transa, transb, m, n, k, &alpha, dArrA, lda, dArrB, ldb, &beta, dArrC, ldc, batch))

    CHECK_CUDA(cudaMemcpy(arrC, dArrC, sizeof(half*) * batch, cudaMemcpyDeviceToHost))

    half *tmp_out;
    CHECK_CUDA(cudaMalloc(&tmp_out, sizeof(half) * batch * m * n))

    for (int i = 0; i < batch; i++) {
        CHECK_CUDA(cudaMemcpy(tmp_out + i * m * n, arrC[i], sizeof(half) * m * n, cudaMemcpyDeviceToDevice));
    }

    CHECK_CUBLAS( cublasScalEx(handle, batch * m * n, &scale, CUDA_R_32F, tmp_out, CUDA_R_16F, 1, CUDA_R_32F) )

    dim3 grid(batch / 32 + 1, m, n);
    transpose_batches<<<grid, 32>>>(tmp_out, output, batch, m, n);

//    for (int i = 0; i < batch; i++) {
//        half *c_out = new half[m * n];
//        cudaMemcpy(c_out, arrC[i], sizeof(half) * m * k, cudaMemcpyDeviceToHost);
//        //cudaMemcpy(c_out, dArrC + i, sizeof(half) * m * n, cudaMemcpyDeviceToHost);
//        for (int j = 0; j < m; j++) {
//            for (int v = 0; v < n; v++) {
//                printf("%d ", __half2int_rz(c_out[j * n + v]));
//            }
//            printf("\n");
//        }
//    }
    /* free resources */
    CHECK_CUDA( cudaFree(d_C) );
    CHECK_CUDA( cudaFree(dArrA) );
    CHECK_CUDA( cudaFree(dArrB) );
    CHECK_CUDA( cudaFree(dArrC) );
    CHECK_CUBLAS( cublasDestroy(handle) );

}

void cublas_gemm_batches_scale_device_v2(half *d_A, half *d_B, int batch, int inputM, int inputK, int inputN,
                                         float scale, half *output) {
    // 因为为列存储，为了方便，设置转置
    cublasHandle_t cublasH = nullptr;

    const int m = inputM;
    const int n = inputN;
    const int k = inputK;
    const int lda = k; // 因为转置了 因此ld代表列数
    const int ldb = n;
    const int ldc = m; // c的ld都是m

    const half alpha = 1.0;
    const half beta = 0.0;
    int64_t strideA = m * k, strideB = k * n, strideC = m * n;

    half *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_T;

    /* step 1: create cublas handle, bind a stream */
    CHECK_CUBLAS( cublasCreate(&cublasH) );

    /* step 2: copy data to device */
    CHECK_CUDA( cudaMalloc(&d_C, sizeof(half) * batch * m * n) );

    /* step 3: compute */
    CHECK_CUBLAS( cublasHgemmStridedBatched(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, strideA,
                                            d_B, ldb, strideB, &beta, d_C, ldc, strideC, batch) );

    CHECK_CUBLAS( cublasScalEx(cublasH, batch * m * n, &scale, CUDA_R_32F, d_C, CUDA_R_16F, 1, CUDA_R_32F) )

    // transpose
    dim3 grid(batch / 32 + 1, m, n);
    transpose_batches<<<grid, 32>>>(d_C, output, batch, m, n);

    /* step 4: copy data to host */
    //CHECK_CUDA( cudaMemcpyAsync(output, d_C, sizeof(half) * m * n, cudaMemcpyDeviceToDevice, stream));

    /* free resources */
    CHECK_CUDA( cudaFree(d_C) );
    CHECK_CUBLAS( cublasDestroy(cublasH) );
}