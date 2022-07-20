//
// Created by dbettkk on 2022/7/14.
//

#include "sparse_mma_gemm.cuh"

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
    CHECK_CUDA(cudaMalloc((void **) &dA, A_size))
    CHECK_CUDA(cudaMalloc((void **) &dB, B_size))
    CHECK_CUDA(cudaMalloc((void **) &dC, C_size))
    CHECK_CUDA(cudaMalloc((void **) &d_valid, sizeof(d_valid)))
    CHECK_CUDA(cudaMemset(dC, 0, C_size))
    dD = dC;

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
            std::printf("!!!! The matrix need to be pruned. valid: %d\n", *is_valid);
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
                            cudaMemcpyDeviceToHost))

    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )

}

void cublas_gemm(const half *inputA, const half *inputB, int inputM, int inputK, int inputN, half *output) {
    // 因为为列存储，为了方便，设置转置
    cublasHandle_t cublasH = nullptr;
    cudaStream_t stream = nullptr;

    const int m = inputM;
    const int n = inputN;
    const int k = inputK;
    const int lda = k; // 因为转置了 因此ld代表列数
    const int ldb = n;
    const int ldc = m; // c的ld都是m

    const half alpha = 1.0;
    const half beta = 0.0;

    half *d_A = nullptr;
    half *d_B = nullptr;
    half *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_T;

    /* step 1: create cublas handle, bind a stream */
    CHECK_CUBLAS( cublasCreate(&cublasH) );

    CHECK_CUDA( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
    CHECK_CUBLAS( cublasSetStream(cublasH, stream) );

    /* step 2: copy data to device */
    CHECK_CUDA( cudaMalloc(&d_A, sizeof(half) * m * k) );
    CHECK_CUDA( cudaMalloc(&d_B, sizeof(half) * k * n) );
    CHECK_CUDA( cudaMalloc(&d_C, sizeof(half) * m * n) );

    CHECK_CUDA( cudaMemcpyAsync(d_A, inputA, sizeof(half) * m * k, cudaMemcpyHostToDevice, stream) );
    CHECK_CUDA( cudaMemcpyAsync(d_B, inputB, sizeof(half) * k * n, cudaMemcpyHostToDevice, stream) );


    /* step 3: compute */
    CHECK_CUBLAS( cublasHgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc) );

    /* step 4: copy data to host */
    CHECK_CUDA( cudaMemcpyAsync(output, d_C, sizeof(half) * m * n, cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA( cudaStreamSynchronize(stream) );

    /* free resources */
    CHECK_CUDA( cudaFree(d_A) );
    CHECK_CUDA( cudaFree(d_B) );
    CHECK_CUDA( cudaFree(d_C) );

    CHECK_CUBLAS( cublasDestroy(cublasH) );

    CHECK_CUDA( cudaStreamDestroy(stream) );

    //CHECK_CUDA( cudaDeviceReset() );
}

void cublas_gemm_device(const half *d_A, const half *d_B, int inputM, int inputK, int inputN, half *output) {
    // 因为为列存储，为了方便，设置转置
    cublasHandle_t cublasH = nullptr;
    cudaStream_t stream = nullptr;

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

    CHECK_CUDA( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
    CHECK_CUBLAS( cublasSetStream(cublasH, stream) );

    /* step 2: copy data to device */
    CHECK_CUDA( cudaMalloc(&d_C, sizeof(half) * m * n) );

    /* step 3: compute */
    CHECK_CUBLAS( cublasHgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc) );

    /* step 4: copy data to host */
    CHECK_CUDA( cudaMemcpyAsync(output, d_C, sizeof(half) * m * n, cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA( cudaStreamSynchronize(stream) );

    /* free resources */
    //CHECK_CUDA( cudaFree(d_A) );
    //CHECK_CUDA( cudaFree(d_B) );
    CHECK_CUDA( cudaFree(d_C) );

    CHECK_CUBLAS( cublasDestroy(cublasH) );

    CHECK_CUDA( cudaStreamDestroy(stream) );

    //CHECK_CUDA( cudaDeviceReset() );
}

__global__ void transpose1(half *A, half *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[nx * N + ny] = A[ny * N + nx];
    }
}

__global__ void transpose2(half *A, half *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = A[nx * N + ny];
    }
}

__global__ void mask_matrix_gpu(half *tgt, const int *mask_mat, int row, int col) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < row * col) {
        if (mask_mat[idx] == 0) tgt[idx] = 0;
    }
}

void position_encoding(half *input, int batch, int max_sen_len, int ebd) {
    return;
}

void softmax(half *item) {
    return;
}

__global__ void transpose(half *src, half* tgt, int row, int col) {
    return;
}