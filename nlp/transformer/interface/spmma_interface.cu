//
// Created by dbettkk on 2022/8/14.
//

#include "spmma_interface.cuh"

void padCudaMemcpy2D(const half* src, int row, int col, half *dest, int row_padding, int col_padding) {
    CHECK_CUDA( cudaMemset(dest, 0, row_padding * col_padding * sizeof(half)) )
    if (col == col_padding) {
        //CHECK_CUDA( cudaMemcpy(dest, src, row * col_padding * sizeof(half), cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(dest, src, row * col_padding * sizeof(half), cudaMemcpyDeviceToDevice) )
    } else {
        // spitch指定想要复制的矩阵的本身的宽度 width指定需要复制的宽度 dpitch指定赋值到dest的宽度
        //CHECK_CUDA( cudaMemcpy2D(dest, col_padding * sizeof(half), src, col * sizeof(half), col * sizeof(half), row, cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy2D(dest, col_padding * sizeof(half), src, col * sizeof(half), col * sizeof(half), row, cudaMemcpyDeviceToDevice) )
    }
}

void sparse_mma_gemm_device(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, bool isValid, half *outputD) {
    int m = inputM % 8 ? inputM + 8 - inputM % 8 : inputM;
    int k = inputK % 16 ? inputK + 16 - inputK % 16 : inputK;
    int n = inputN % 8 ? inputN + 8 - inputN % 8 : inputN;

    size_t A_size = m * k * sizeof(half);
    size_t B_size = k * n * sizeof(half);
    size_t C_size = m * n * sizeof(half);
    // device
    half *dA, *dB, *dC, *dD, *dA_compressed;
    int *d_valid;
    int *is_valid = (int *)malloc(sizeof(int));

    CHECK_CUDA(cudaMalloc((void **) &dC, C_size))
    CHECK_CUDA(cudaMalloc((void **) &d_valid, sizeof(d_valid)))
    CHECK_CUDA(cudaMemset(dC, 0, C_size))
    dD = dC;

    CHECK_CUDA(cudaMalloc((void **) &dA, A_size))
    CHECK_CUDA(cudaMalloc((void **) &dB, B_size))
    // padding to match mma.sp
    padCudaMemcpy2D(inputA, inputM, inputK, dA, m, k);
    padCudaMemcpy2D(inputB, inputK, inputN, dB, k, n);

    // Leading dimension 如果行优先则代表列数
    int lda = k, ldb = n, ldc = n;
    auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto order = CUSPARSE_ORDER_ROW; // cusparseOrder_t
    auto type = CUDA_R_16F;
    auto compute_type = CUSPARSE_COMPUTE_16F;
    float alpha = 1.0f;
    float beta = 0.0f;
    unsigned alignment = 16;

    //--------------------------------------------------------------------------

    cusparseLtHandle_t handle;
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    cudaStream_t stream = nullptr;
    CHECK_CUSPARSE(cusparseLtInit(&handle))
    // matrix descriptor initialization
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matA, m, k, lda, alignment, type, order,
                                                      CUSPARSELT_SPARSITY_50_PERCENT))
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, k, n, ldb, alignment, type, order))
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC, m, n, ldc, alignment, type, order))
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type))
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
    int alg = 0;    // 算法
    CHECK_CUSPARSE(
            cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))

    size_t workspace_size, compressed_size;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size))
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size))

    //--------------------------------------------------------------------------
    // Prune and Compress
    if (!isValid) {
        // 不符合条件 需要进行剪枝
        //int is_valid = 0;
        CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream))
        CHECK_CUDA(cudaMemcpyAsync(is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream))
        CHECK_CUDA(cudaStreamSynchronize(stream))
        if (*is_valid == 1) {
            //if (!check_sparse(dA, m, k)) printf("no fit\n");
            //else printf("fit\n");
            printf("!!!! The matrix need to be pruned. valid: %d\n", *is_valid);
            CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream))
        }
    }
    // 符合条件 不用判断 直接compress即可
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size))
    CHECK_CUDA(cudaMalloc((void **) &dA_compressed, compressed_size))
    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream))

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size))

    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, streams,
                                    num_streams))

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // destroy plan and handle
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC))
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
    CHECK_CUSPARSE(cusparseLtDestroy(&handle))
    //--------------------------------------------------------------------------

    // padding后的fix
    CHECK_CUDA(cudaMemcpy2D(outputD, inputN * sizeof(half), dD, n * sizeof(half), inputN * sizeof(half), inputM,
                            cudaMemcpyDeviceToDevice))

    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )

}

void sparse_mma_gemm_noPad_device(half *inputA, half *inputB, int inputM, int inputK, int inputN, bool isValid, half *outputD) {
    int m = inputM;
    int k = inputK;
    int n = inputN;

    size_t C_size = m * n * sizeof(half);
    // device
    half *dC, *dD, *dA_compressed;
    int *d_valid;
    int *is_valid = (int *)malloc(sizeof(int));

    CHECK_CUDA(cudaMalloc((void **) &dC, C_size))
    CHECK_CUDA(cudaMalloc((void **) &d_valid, sizeof(d_valid)))
    CHECK_CUDA(cudaMemset(dC, 0, C_size))
    dD = dC;

    // Leading dimension 如果行优先则代表列数
    int lda = k, ldb = n, ldc = n;
    auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto order = CUSPARSE_ORDER_ROW; // cusparseOrder_t
    auto type = CUDA_R_16F;
    auto compute_type = CUSPARSE_COMPUTE_16F;
    float alpha = 1.0f;
    float beta = 0.0f;
    unsigned alignment = 16;

    //--------------------------------------------------------------------------

    cusparseLtHandle_t handle;
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    cudaStream_t stream = nullptr;
    CHECK_CUSPARSE(cusparseLtInit(&handle))
    // matrix descriptor initialization
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matA, m, k, lda, alignment, type, order,
                                                      CUSPARSELT_SPARSITY_50_PERCENT))
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, k, n, ldb, alignment, type, order))
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC, m, n, ldc, alignment, type, order))
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type))
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
    int alg = 0;    // 算法
    CHECK_CUSPARSE(
            cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))

    size_t workspace_size, compressed_size;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size))
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size))

    //--------------------------------------------------------------------------
    // Prune and Compress
    if (!isValid) {
        // 不符合条件 需要进行剪枝
        //int is_valid = 0;
        CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, inputA, d_valid, stream))
        CHECK_CUDA(cudaMemcpyAsync(is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream))
        CHECK_CUDA(cudaStreamSynchronize(stream))
        if (*is_valid == 1) {
            //if (!check_sparse(dA, m, k)) printf("no fit\n");
            //else printf("fit\n");
            printf("!!!! The matrix need to be pruned. valid: %d\n", *is_valid);
            CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, inputA, inputA, CUSPARSELT_PRUNE_SPMMA_TILE, stream))
        }
    }
    // 符合条件 不用判断 直接compress即可
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size))
    CHECK_CUDA(cudaMalloc((void **) &dA_compressed, compressed_size))
    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, inputA, dA_compressed, stream))

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size))

    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, inputB, &beta, dC, outputD, d_workspace, streams,
                                    num_streams))

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // destroy plan and handle
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC))
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
    CHECK_CUSPARSE(cusparseLtDestroy(&handle))
    //--------------------------------------------------------------------------
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )
}

void sparse_mma_gemm_batches_device(const half *inputA, const half *inputB, int batch, int inputM, int inputK, int inputN, bool isValid, half *outputD) {
    int m = inputM % 8 ? inputM + 8 - inputM % 8 : inputM;
    int k = inputK % 16 ? inputK + 16 - inputK % 16 : inputK;
    int n = inputN % 8 ? inputN + 8 - inputN % 8 : inputN;

    size_t A_size = batch * m * k * sizeof(half);
    size_t B_size = batch * k * n * sizeof(half);
    size_t C_size = batch * m * n * sizeof(half);
    // device
    half *dA, *dB, *dC, *dD, *dA_compressed;
    int *d_valid;
    int *is_valid = (int *)malloc(sizeof(int));

    CHECK_CUDA(cudaMalloc((void **) &dC, C_size))
    CHECK_CUDA(cudaMalloc((void **) &d_valid, sizeof(d_valid)))
    CHECK_CUDA(cudaMemset(dC, 0, C_size))
    dD = dC;

    //auto ttt = new CudaTime();
    //ttt->initAndStart();
    CHECK_CUDA(cudaMalloc((void **) &dA, A_size))
    CHECK_CUDA(cudaMalloc((void **) &dB, B_size))
    // padding to match mma.sp
    for (int i = 0; i < batch; i++) {
        padCudaMemcpy2D(inputA + i * inputM * inputK, inputM, inputK, dA + i * m * k, m, k);
        padCudaMemcpy2D(inputB + i * inputK * inputN, inputK, inputN, dB + i * n * k, k, n);
    }
    //printf("pad time: %fms\t", ttt->endAndGetTime());
    // Leading dimension 如果行优先则代表列数
    int lda = k, ldb = n, ldc = n;
    auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto order = CUSPARSE_ORDER_ROW; // cusparseOrder_t
    auto type = CUDA_R_16F;
    auto compute_type = CUSPARSE_COMPUTE_16F;
    float alpha = 1.0f;
    float beta = 0.0f;
    unsigned alignment = 16;

    //--------------------------------------------------------------------------

    cusparseLtHandle_t handle;
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    cudaStream_t stream = nullptr;
    CHECK_CUSPARSE(cusparseLtInit(&handle))
    // matrix descriptor initialization
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matA, m, k, lda, alignment, type, order,
                                                      CUSPARSELT_SPARSITY_50_PERCENT))
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, k, n, ldb, alignment, type, order))
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC, m, n, ldc, alignment, type, order))
    // batch
    int64_t batch_strideA = m * k, batch_strideB = n * k, batch_strideC = m * n;
    CHECK_CUSPARSE(cusparseLtMatDescSetAttribute(&handle, &matA, CUSPARSELT_MAT_NUM_BATCHES, &batch, sizeof(batch)))
    CHECK_CUSPARSE(cusparseLtMatDescSetAttribute(&handle, &matB, CUSPARSELT_MAT_NUM_BATCHES, &batch, sizeof(batch)))
    CHECK_CUSPARSE(cusparseLtMatDescSetAttribute(&handle, &matC, CUSPARSELT_MAT_NUM_BATCHES, &batch, sizeof(batch)))
    CHECK_CUSPARSE(cusparseLtMatDescSetAttribute(&handle, &matA, CUSPARSELT_MAT_BATCH_STRIDE, &batch_strideA, sizeof(batch_strideA)))
    CHECK_CUSPARSE(cusparseLtMatDescSetAttribute(&handle, &matB, CUSPARSELT_MAT_BATCH_STRIDE, &batch_strideB, sizeof(batch_strideB)))
    CHECK_CUSPARSE(cusparseLtMatDescSetAttribute(&handle, &matC, CUSPARSELT_MAT_BATCH_STRIDE, &batch_strideC, sizeof(batch_strideC)))
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type))
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
    int alg = 0;    // 算法
    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
    size_t workspace_size, compressed_size;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size))
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size))
    //--------------------------------------------------------------------------
    // Prune and Compress
    if (!isValid) {
        // 不符合条件 需要进行剪枝
        CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream))
        CHECK_CUDA(cudaMemcpyAsync(is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream))
        CHECK_CUDA(cudaStreamSynchronize(stream))
        if (*is_valid == 1) {
            printf("!!!! The matrix need to be pruned. valid: %d\n", *is_valid);
            CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream))
        }
    }
    // 符合条件 不用判断 直接compress即可
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size))
    CHECK_CUDA(cudaMalloc((void **) &dA_compressed, compressed_size))
    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream))

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size))

    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, streams,
                                    num_streams))

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // destroy plan and handle
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC))
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
    CHECK_CUSPARSE(cusparseLtDestroy(&handle))
    //--------------------------------------------------------------------------

    // padding后的fix
    for (int i = 0; i < batch; i++) {
        CHECK_CUDA(cudaMemcpy2D(outputD + i * inputM * inputN, inputN * sizeof(half), dD + i * m * n, n * sizeof(half),
                                inputN * sizeof(half), inputM, cudaMemcpyDeviceToDevice))
    }
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )
}

void sparse_mma_gemm_noPad_batches_device(half *inputA, half *inputB, int batch, int inputM, int inputK, int inputN, bool isValid, half *outputD) {
    int m = inputM, k = inputK, n = inputN;

    size_t A_size = batch * m * k * sizeof(half);
    size_t B_size = batch * k * n * sizeof(half);
    size_t C_size = batch * m * n * sizeof(half);
    // device
    half *dC, *dD, *dA_compressed;
    int *d_valid;
    int *is_valid = (int *)malloc(sizeof(int));

    CHECK_CUDA(cudaMalloc((void **) &dC, C_size))
    CHECK_CUDA(cudaMalloc((void **) &d_valid, sizeof(d_valid)))
    CHECK_CUDA(cudaMemset(dC, 0, C_size))
    dD = dC;

    // Leading dimension 如果行优先则代表列数
    int lda = k, ldb = n, ldc = n;
    auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto order = CUSPARSE_ORDER_ROW; // cusparseOrder_t
    auto type = CUDA_R_16F;
    auto compute_type = CUSPARSE_COMPUTE_16F;
    float alpha = 1.0f;
    float beta = 0.0f;
    unsigned alignment = 16;

    cusparseLtHandle_t handle;
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    cudaStream_t stream = nullptr;
    CHECK_CUSPARSE(cusparseLtInit(&handle))
    // matrix descriptor initialization
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matA, m, k, lda, alignment, type, order,
                                                      CUSPARSELT_SPARSITY_50_PERCENT))
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, k, n, ldb, alignment, type, order))
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC, m, n, ldc, alignment, type, order))
    // batch
    int64_t batch_strideA = m * k, batch_strideB = n * k, batch_strideC = m * n;
    CHECK_CUSPARSE(cusparseLtMatDescSetAttribute(&handle, &matA, CUSPARSELT_MAT_NUM_BATCHES, &batch, sizeof(batch)))
    CHECK_CUSPARSE(cusparseLtMatDescSetAttribute(&handle, &matB, CUSPARSELT_MAT_NUM_BATCHES, &batch, sizeof(batch)))
    CHECK_CUSPARSE(cusparseLtMatDescSetAttribute(&handle, &matC, CUSPARSELT_MAT_NUM_BATCHES, &batch, sizeof(batch)))
    CHECK_CUSPARSE(cusparseLtMatDescSetAttribute(&handle, &matA, CUSPARSELT_MAT_BATCH_STRIDE, &batch_strideA, sizeof(batch_strideA)))
    CHECK_CUSPARSE(cusparseLtMatDescSetAttribute(&handle, &matB, CUSPARSELT_MAT_BATCH_STRIDE, &batch_strideB, sizeof(batch_strideB)))
    CHECK_CUSPARSE(cusparseLtMatDescSetAttribute(&handle, &matC, CUSPARSELT_MAT_BATCH_STRIDE, &batch_strideC, sizeof(batch_strideC)))
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type))
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
    int alg = 0;    // 算法
    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
    size_t workspace_size, compressed_size;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size))
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size))
    //--------------------------------------------------------------------------
    // Prune and Compress
    if (!isValid) {
        // 不符合条件 需要进行剪枝
        CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, inputA, d_valid, stream))
        CHECK_CUDA(cudaMemcpyAsync(is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream))
        CHECK_CUDA(cudaStreamSynchronize(stream))
        if (*is_valid == 1) {
            printf("!!!! The matrix need to be pruned. valid: %d\n", *is_valid);
            CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, inputA, inputA, CUSPARSELT_PRUNE_SPMMA_TILE, stream))
        }
    }
    // 符合条件 不用判断 直接compress即可
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size))
    CHECK_CUDA(cudaMalloc((void **) &dA_compressed, compressed_size))
    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, inputA, dA_compressed, stream))

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size))

    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, inputB, &beta, dC, outputD, d_workspace, streams,
                                    num_streams))

    // destroy plan and handle
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC))
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
    CHECK_CUSPARSE(cusparseLtDestroy(&handle))
    //--------------------------------------------------------------------------
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )
}

/** v0.3.0 */
//void sparse_mma_gemm_device_v2(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, bool isValid, half *outputD) {
//    int m = inputM % 8 ? inputM + 8 - inputM % 8 : inputM;
//    int k = inputK % 16 ? inputK + 16 - inputK % 16 : inputK;
//    int n = inputN % 8 ? inputN + 8 - inputN % 8 : inputN;
//
//    size_t A_size = m * k * sizeof(half);
//    size_t B_size = k * n * sizeof(half);
//    size_t C_size = m * n * sizeof(half);
//    // device
//    half *dA, *dB, *dC, *dD, *dA_compressed;
//    int *d_valid;
//    int *is_valid = (int *)malloc(sizeof(int));
//
//    CHECK_CUDA(cudaMalloc((void **) &dC, C_size))
//    CHECK_CUDA(cudaMalloc((void **) &d_valid, sizeof(d_valid)))
//    CHECK_CUDA(cudaMemset(dC, 0, C_size))
//    dD = dC;
//
//    //auto ttt = new CudaTime();
//    //ttt->initAndStart();
//    CHECK_CUDA(cudaMalloc((void **) &dA, A_size))
//    CHECK_CUDA(cudaMalloc((void **) &dB, B_size))
//    // padding to match mma.sp
//    padCudaMemcpy2D(inputA, inputM, inputK, dA, m, k);
//    padCudaMemcpy2D(inputB, inputK, inputN, dB, k, n);
//    //printf("pad time: %fms\t", ttt->endAndGetTime());
//    // Leading dimension 如果行优先则代表列数
//    int lda = k, ldb = n, ldc = n;
//    auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
//    auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
//    auto order = CUSPARSE_ORDER_ROW; // cusparseOrder_t
//    auto type = CUDA_R_16F;
//    auto compute_type = CUSPARSE_COMPUTE_16F;
//    float alpha = 1.0f;
//    float beta = 0.0f;
//    unsigned alignment = 16;
//
//    //--------------------------------------------------------------------------
//
//    cusparseLtHandle_t handle;
//    cusparseLtMatDescriptor_t matA, matB, matC;
//    cusparseLtMatmulDescriptor_t matmul;
//    cusparseLtMatmulAlgSelection_t alg_sel;
//    cusparseLtMatmulPlan_t plan;
//    cudaStream_t stream = nullptr;
//    CHECK_CUSPARSE(cusparseLtInit(&handle))
//    // matrix descriptor initialization
//    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matA, m, k, lda, alignment, type, order,
//                                                      CUSPARSELT_SPARSITY_50_PERCENT))
//    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, k, n, ldb, alignment, type, order))
//    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC, m, n, ldc, alignment, type, order))
//    // matmul, algorithm selection, and plan initialization
//    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type))
//    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
//    int alg = 0;
//    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
//    size_t workspace_size = 0, compressed_size;
//    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size) )
//    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size))
//
//    // Prune and Compress
//    if (!isValid) {
//        // 不符合条件 需要进行剪枝
//        //int is_valid = 0;
//        CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream))
//        CHECK_CUDA(cudaMemcpyAsync(is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream))
//        CHECK_CUDA(cudaStreamSynchronize(stream))
//        if (*is_valid == 1) {
//            //if (!check_sparse(dA, m, k)) printf("no fit\n");
//            //else printf("fit\n");
//            printf("!!!! The matrix need to be pruned. valid: %d\n", *is_valid);
//            CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream))
//        }
//    }
//    // 符合条件 不用判断 直接compress即可
//    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size))
//    CHECK_CUDA(cudaMalloc((void **) &dA_compressed, compressed_size))
//    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream))
//
//    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//    // Search the best kernel
//    void*         d_workspace = nullptr;
//    int           num_streams = 0;
//    cudaStream_t* streams     = nullptr;
//
//    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size))
//
//    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, streams,
//                                    num_streams))
//
//    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//    // destroy plan and handle
//    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA))
//    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB))
//    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC))
//    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
//    CHECK_CUSPARSE(cusparseLtDestroy(&handle))
//    //--------------------------------------------------------------------------
//
//    // padding后的fix
//    CHECK_CUDA(cudaMemcpy2D(outputD, inputN * sizeof(half), dD, n * sizeof(half), inputN * sizeof(half), inputM,
//                            cudaMemcpyDeviceToHost))
//
//    CHECK_CUDA( cudaFree(dA_compressed) )
//    CHECK_CUDA( cudaFree(dA) )
//    CHECK_CUDA( cudaFree(dB) )
//    CHECK_CUDA( cudaFree(dC) )
//    CHECK_CUDA( cudaFree(d_valid) )
//
//}
//
//void sparse_mma_gemm_splitK_device(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, bool isValid, half *outputD) {
//    int m = inputM % 8 ? inputM + 8 - inputM % 8 : inputM;
//    int k = inputK % 16 ? inputK + 16 - inputK % 16 : inputK;
//    int n = inputN % 8 ? inputN + 8 - inputN % 8 : inputN;
//
//    size_t A_size = m * k * sizeof(half);
//    size_t B_size = k * n * sizeof(half);
//    size_t C_size = m * n * sizeof(half);
//    // device
//    half *dA, *dB, *dC, *dD, *dA_compressed;
//    int *d_valid;
//    int *is_valid = (int *)malloc(sizeof(int));
//
//    CHECK_CUDA(cudaMalloc((void **) &dC, C_size))
//    CHECK_CUDA(cudaMalloc((void **) &d_valid, sizeof(d_valid)))
//    CHECK_CUDA(cudaMemset(dC, 0, C_size))
//    dD = dC;
//
//    //auto ttt = new CudaTime();
//    //ttt->initAndStart();
//    CHECK_CUDA(cudaMalloc((void **) &dA, A_size))
//    CHECK_CUDA(cudaMalloc((void **) &dB, B_size))
//    // padding to match mma.sp
//    padCudaMemcpy2D(inputA, inputM, inputK, dA, m, k);
//    padCudaMemcpy2D(inputB, inputK, inputN, dB, k, n);
//    //printf("pad time: %fms\t", ttt->endAndGetTime());
//    // Leading dimension 如果行优先则代表列数
//    int lda = k, ldb = n, ldc = n;
//    auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
//    auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
//    auto order = CUSPARSE_ORDER_ROW; // cusparseOrder_t
//    auto type = CUDA_R_16F;
//    auto compute_type = CUSPARSE_COMPUTE_16F;
//    float alpha = 1.0f;
//    float beta = 0.0f;
//    unsigned alignment = 16;
//
//    //--------------------------------------------------------------------------
//
//    cusparseLtHandle_t handle;
//    cusparseLtMatDescriptor_t matA, matB, matC;
//    cusparseLtMatmulDescriptor_t matmul;
//    cusparseLtMatmulAlgSelection_t alg_sel;
//    cusparseLtMatmulPlan_t plan;
//    cudaStream_t stream = nullptr;
//    CHECK_CUSPARSE(cusparseLtInit(&handle))
//    // matrix descriptor initialization
//    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matA, m, k, lda, alignment, type, order,
//                                                      CUSPARSELT_SPARSITY_50_PERCENT))
//    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, k, n, ldb, alignment, type, order))
//    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC, m, n, ldc, alignment, type, order))
//    // matmul, algorithm selection, and plan initialization
//    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type))
//    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
//    int alg = 0;
//    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
//    // Split-K Mode
//
//    int splitK = 4, splitKBuffers = 1;
//    cusparseLtSplitKMode_t splitKMode = CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS;
//    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K, &splitK, sizeof(splitK)) )
//    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K_MODE, &splitKMode, sizeof(splitKMode)) )
//    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K_BUFFERS, &splitKBuffers, sizeof(splitKBuffers)) )
//    size_t workspace_size = 0, compressed_size;
//    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size) )
//    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size))
//
////
////    auto mode = splitKMode == CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL ? "ONE_KERNEL"  : splitKMode == CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS ? "TWO_KERNELS" : "invalid";
////    printf("splitK=%d, splitK-mode=%d, splitK-buffers=%d\n\n", splitK, splitKMode, splitKBuffers);
//
//    // Prune and Compress
//    if (!isValid) {
//        // 不符合条件 需要进行剪枝
//        //int is_valid = 0;
//        CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream))
//        CHECK_CUDA(cudaMemcpyAsync(is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream))
//        CHECK_CUDA(cudaStreamSynchronize(stream))
//        if (*is_valid == 1) {
//            //if (!check_sparse(dA, m, k)) printf("no fit\n");
//            //else printf("fit\n");
//            printf("!!!! The matrix need to be pruned. valid: %d\n", *is_valid);
//            CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream))
//        }
//    }
//    // 符合条件 不用判断 直接compress即可
//    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size))
//    CHECK_CUDA(cudaMalloc((void **) &dA_compressed, compressed_size))
//    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream))
//
//    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//    // Search the best kernel
//    void*         d_workspace = nullptr;
//    int           num_streams = 0;
//    cudaStream_t* streams     = nullptr;
//
//    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size))
//
////    CHECK_CUSPARSE(cusparseLtMatmulSearch(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, streams,
////                           num_streams))
////
////    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K, &splitK, sizeof(splitK)) )
////
////    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K_MODE, &splitKMode, sizeof(splitKMode)) )
////
////    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K_BUFFERS, &splitKBuffers, sizeof(splitKBuffers)) )
////    auto mode = splitKMode == CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL ? "ONE_KERNEL"  : splitKMode == CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS ? "TWO_KERNELS" : "invalid";
////    printf("splitK=%d, splitK-mode=%d, splitK-buffers=%d\n\n", splitK, splitKMode, splitKBuffers);
//
//    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, streams,
//                                    num_streams))
//
//    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//    // destroy plan and handle
//    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA))
//    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB))
//    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC))
//    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
//    CHECK_CUSPARSE(cusparseLtDestroy(&handle))
//    //--------------------------------------------------------------------------
//
//    // padding后的fix
//    CHECK_CUDA(cudaMemcpy2D(outputD, inputN * sizeof(half), dD, n * sizeof(half), inputN * sizeof(half), inputM,
//                            cudaMemcpyDeviceToHost))
//
//    CHECK_CUDA( cudaFree(dA_compressed) )
//    CHECK_CUDA( cudaFree(dA) )
//    CHECK_CUDA( cudaFree(dB) )
//    CHECK_CUDA( cudaFree(dC) )
//    CHECK_CUDA( cudaFree(d_valid) )
//
//}
