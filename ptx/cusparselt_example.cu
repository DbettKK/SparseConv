//
// Created by dbettkk on 2022/6/16.
//
#include <cusparseLt.h>
#include <cuda_fp16.h>
#include <iostream>
#include "wmma.sp.cuh"

void cusparselt_example(half *dA, half *dB, half *dC, half *dD, int m, int k, int n) {
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

    half *dA_compressed;
    int *d_valid;
    cudaMalloc(&d_valid, sizeof(d_valid));
    //--------------------------------------------------------------------------

    cusparseLtHandle_t handle;
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    cudaStream_t stream = nullptr;
    cusparseLtInit(&handle);
    // matrix descriptor initialization
    cusparseLtStructuredDescriptorInit(&handle, &matA, m, k, lda, alignment, type, order, CUSPARSELT_SPARSITY_50_PERCENT);
    cusparseLtDenseDescriptorInit(&handle, &matB, k, n, ldb, alignment, type, order);
    cusparseLtDenseDescriptorInit(&handle, &matC, m, n, ldc, alignment, type, order);
    // matmul, algorithm selection, and plan initialization
    cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type);
    cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    int alg = 0;    // 算法
    cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg));

    size_t workspace_size, compressed_size;
    cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size);

    cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size);
    //--------------------------------------------------------------------------
    // Prune and Compress
    int *is_valid = (int *)malloc(sizeof(int));

    // 不符合条件 需要进行剪枝
    //int is_valid = 0;
    cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream);
    cudaMemcpyAsync(is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (*is_valid == 1) {
        std::printf("!!!! The matrix need to be pruned. valid: %d\n", *is_valid);
        cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream);
    }

    // 符合条件 不用判断 直接compress即可
    cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size);
    cudaMalloc((void **) &dA_compressed, compressed_size);
    cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    /*
    int alg_id;
     cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id); )
    printf("best alg: %d\n", alg_id);
    */
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Perform the matrix multiplication

    // time
    auto time = new CudaTime();
    time->initAndStart();

    cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, streams,
                                    num_streams);

    float totalTime = time->endAndGetTime();
    printf("cusparselt calculate took %fms\n", totalTime);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // destroy plan and handle
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtDestroy(&handle);
}

int main9() {
    auto generator = new DataGenerator();
    half *hA = generator->generateSimpleSparse(16, 16, true);
    half *hB = generator->generateNumber(16, 16, 1);
    half *hC = generator->generateZero(16, 16);

    half *dA, *dB, *dC, *dD;
    cudaMalloc(&dA, sizeof(half) * 16 * 16);
    cudaMalloc(&dB, sizeof(half) * 16 * 16);
    cudaMalloc(&dC, sizeof(half) * 16 * 16);
    cudaMalloc(&dD, sizeof(half) * 16 * 16);
    cudaMemcpy(dA, hA, sizeof(half) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(half) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeof(half) * 256, cudaMemcpyHostToDevice);

    cusparselt_example(dA, dB, dC, dD, 16, 16, 16);

    half *hD = new half[256];
    cudaMemcpy(hD, dD, sizeof(half) * 256, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%d ", __half2int_rz(hD[i * 16 + j]));
        }
        printf("\n");
    }

    return 0;
}
