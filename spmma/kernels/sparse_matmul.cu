//
// Created by dbettkk on 2022/3/30.
//
#include"sparse_matmul.cuh"

void
spmma_matmul(const float *inputA, const float *inputB, int inputM, int inputK, int inputN, bool isValid, float *outputD,
             MatrixParam *retParam) {
    int m = inputM % 8 ? inputM + 8 - inputM % 8 : inputM;
    int k = inputK % 16 ? inputK + 16 - inputK % 16 : inputK;
    int n = inputN % 8 ? inputN + 8 - inputN % 8 : inputN;

    size_t A_size = m * k * sizeof(float);
    size_t B_size = k * n * sizeof(float);
    size_t C_size = m * n * sizeof(float);
    // device
    float *dA, *dB, *dC, *dD, *dA_compressed;
    int *is_valid = (int *)malloc(sizeof(int));
    int *d_valid;
    CHECK_CUDA(cudaMalloc((void **) &dA, A_size))
    CHECK_CUDA(cudaMalloc((void **) &dB, B_size))
    CHECK_CUDA(cudaMalloc((void **) &dC, C_size))
    CHECK_CUDA(cudaMalloc((void **) &d_valid, sizeof(d_valid)))
    dD = dC;

    // padding to match mma.sp
    padMatrix<float>(inputA, inputM, inputK, dA, m, k);
    padMatrix<float>(inputB, inputK, inputN, dB, k, n);
    CHECK_CUDA(cudaMemset(dC, 0, C_size))

    if (!check_sparse(dA, m, k)) printf("not match\n");


    // Leading dimension 如果行优先则代表列数
    int lda = k, ldb = n, ldc = n;
    auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto order = CUSPARSE_ORDER_ROW; // cusparseOrder_t
    auto type = CUDA_R_32F;
    auto compute_type = CUSPARSE_COMPUTE_TF32;
    float alpha = 1.0f;
    float beta = 0.0f;
    unsigned alignment = 16;

    //--------------------------------------------------------------------------
    auto time_all = new CudaTime();
    time_all->initAndStart();

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

    size_t workspace_size;
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
            std::printf("!!!! The matrix need to be pruned.\n");
            CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream))
        }
    }
    // 符合条件 不用判断 直接compress即可
    size_t compressed_size;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size))
    CHECK_CUDA(cudaMalloc((void **) &dA_compressed, compressed_size))

    auto time_cmpr = new CudaTime();
    time_cmpr->initAndStart();
    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream))
    float cmpr_time = time_cmpr->endAndGetTime();
    //printf("compress took: %fms\n", cmpr_time);
    // 对compress后的进行拆分

//    data_cmpr = new float[m * k]; // data部分
//    cudaMemcpy(data_cmpr, dA, m * k * sizeof(float), cudaMemcpyDeviceToHost);
//    printf("2:\n");
//    for (int i = 0; i < m; i++) {
//        for (int j = 0; j < k; j++) {
//            printf("%d ", int(data_cmpr[i * k + j]));
//        }
//        printf("\n");
//    }
    //--------------------------------------------------------------------------

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void *d_workspace = nullptr;
    int num_streams = 0;
    cudaStream_t *streams = nullptr;
    /*
    int alg_id;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)) )
    printf("best alg: %d\n", alg_id);
    */
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Perform the matrix multiplication

    // time
    auto *time = new CudaTime();
    time->initAndStart();

    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, streams,
                                    num_streams))

    float totalTime = time->endAndGetTime();
    printf("cusparselt calculate took %fms\t", totalTime);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // destroy plan and handle
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB))
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC))
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
    CHECK_CUSPARSE(cusparseLtDestroy(&handle))
    //--------------------------------------------------------------------------

    // padding后的fix
    CHECK_CUDA(cudaMemcpy2D(outputD, inputN * sizeof(float), dD, n * sizeof(float), inputN * sizeof(float), inputM,
                            cudaMemcpyDeviceToDevice))

    float all_time = time_all->endAndGetTime();
    printf("spmma all took %fms\n", all_time);

    ofstream out("../data/spmma_time.txt", ios::app);
    out << "spmma_matmul: " << all_time << "ms\n";
    out.close();

    if (retParam != nullptr) {
        retParam->initIfNull();
        // 此时的dC已经不是全0了
        float *tmpC;
        CHECK_CUDA(cudaMalloc((void **) &tmpC, m * n * sizeof(float)))
        CHECK_CUDA(cudaMemset(tmpC, 0, m * n * sizeof(float)))
        retParam->copyFromDevice(dA, dB, tmpC, dD, m, k, n);
        CHECK_CUDA(cudaFree(tmpC))
    }

    CHECK_CUDA(cudaFree(dA))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))
    CHECK_CUDA(cudaFree(d_valid))

}


