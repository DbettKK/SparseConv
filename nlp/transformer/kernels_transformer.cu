//
// Created by dbettkk on 2022/7/25.
//
#include "kernels_transformer.cuh"

bool check_sparse(half *item, int row, int col) {
    printf("m: %d, k: %d\n", row, col);
    half *host = new half[row * col];
    cudaMemcpy(host, item, sizeof(half) * row * col, cudaMemcpyDeviceToHost);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j+=4) {
            int zero_cnt = 0;
            for (int start = 0; start < 4; start++) {
                if (__half2float(host[i * col + j + start]) == 0) {
                    zero_cnt++;
                }
            }
            if (zero_cnt < 2) {
                return false;
            }
        }
    }
    return true;
}

__global__ void softmax_half(half *item, const int row, const int col) {
    __shared__ half mem[64][256]; // 记录每一列
    const int blx = blockIdx.x;
    const int thx = threadIdx.x;
    if (thx < row && blx < col) {
        mem[thx][blx] = item[thx * col + blx]; // 传入
    }
    __syncthreads();
    if (thx < row && blx < col) {
        half max = -65504;
        for (int i = 0; i < row; i++) {
            if (max <= mem[i][blx]) max = mem[i][blx];
        }
        double sum = 0;
        for (int i = 0; i < row; i++) {
            sum += expf(mem[i][blx] - max);
        }
        item[thx * col + blx] = expf(mem[thx][blx] - max) / sum;
    }
}

__global__ void reshape_multi_head(half *A, half *B, const int row, const int col, const int heads)
{
    const int thx = threadIdx.x, thy = threadIdx.y;
    const int blx = blockIdx.x, bly = blockIdx.y;
    B[blx * row * col + (bly * row + thx) * col / heads + thy] = A[blx * row * col + thx * col + (bly * col / heads + thy)];
}

__global__ void transpose_half(half *item, half *out, int row, int col) {
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < row && ny < col)
    {
        out[ny * row + nx] = item[nx * col + ny];
    }
}

__global__ void gemm_simple(half *A, half *B, int m, int k, int n, half *out) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < m && col < k) {
        for (int i = 0; i < n; i++) {
            out[row * k + col] += A[row * n + i] * B[col + i * k];
        }
    }
}

__global__ void mask_matrix_gpu(half *tgt, const int *mask_mat, int row, int col) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < row * col) {
        if (mask_mat[idx] == 0) tgt[idx] = 0;
    }
}

__global__ void relu_half(half *item, int row, int col) {
    int nx = threadIdx.x + blockDim.x * blockIdx.x;
    int ny = threadIdx.y + blockDim.y * blockIdx.y;
    if (nx < row && ny < col) {
        if (item[nx * col + ny] <= __float2half(0))
            item[nx * col + ny] = 0;
    }
}

__global__ void matrix_add(half *A, half *B, half *C, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        C[idx] = A[idx] +B[idx];
    }
}

__global__ void layerNorm_kernel(half *feature, int batch, int max_len, int size, half *means, half *std, half *out) {
    int blx = blockIdx.x, thx = threadIdx.x;
    out[blx * size + thx] = (float)(feature[blx * size + thx] - means[blx]) / ((float)std[blx] + 0.0001);
}

__global__ void getMeanAndStd(half *feature, int batch, int max_len, int size, half *means, half *std) {
    // 每个线程处理 size个元素
    int blx = blockIdx.x;
    int thx = threadIdx.x;
    float mean = 0, mean_2 = 0;
    for (int i = 0; i < size; i++) {
        float item = feature[blx * max_len * size + thx * size + i];
        mean += item;
        mean_2 += item * item;
    }
    mean /= size;
    mean_2 /= size;
    means[blx * max_len + thx] = mean;
    std[blx * max_len + thx] = sqrt(mean_2 - mean * mean);
}


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

void cublas_gemm_batches_device(const half *d_A, const half *d_B, int batch, int inputM, int inputK, int inputN, half *output) {
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
    CHECK_CUDA( cudaMalloc(&d_C, sizeof(half) * m * n) );

    /* step 3: compute */
    half **arrA, **arrB, **arrC;
    CHECK_CUDA(cudaMalloc(&arrA, sizeof(half*) * batch))
    CHECK_CUDA(cudaMalloc(&arrB, sizeof(half*) * batch))
    CHECK_CUDA(cudaMalloc(&arrC, sizeof(half*) * batch))
    for (int i = 0; i < batch; i++) {
        CHECK_CUDA(cudaMalloc(&arrA[i], sizeof(half) * m * k))
        CHECK_CUDA(cudaMalloc(&arrB[i], sizeof(half) * k * n))
        CHECK_CUDA(cudaMalloc(&arrC[i], sizeof(half) * m * n))
        CHECK_CUDA(cudaMemcpy(arrA[i], d_A + batch * m * k, sizeof(half) * m * k, cudaMemcpyDeviceToDevice))
        CHECK_CUDA(cudaMemcpy(arrB[i], d_B + batch * n * k, sizeof(half) * n * k, cudaMemcpyDeviceToDevice))
        // CHECK_CUDA(cudaMemcpy(arrC[i], d_A + batch * m * k, sizeof(half) * m * k, cudaMemcpyDeviceToDevice))
    }
    CHECK_CUBLAS(cublasHgemmBatched(handle, transa, transb, m, n, k, &alpha, arrA, lda, arrB, ldb, &beta, arrC, ldc, batch))

    // transpose
//    dim3 grid(m / 32 + 1, n / 32 + 1);
//    dim3 block(32, 32);
//    transpose_half<<<grid, block>>>(d_C, output, m, n);

    /* step 4: copy data to host */
    //CHECK_CUDA( cudaMemcpyAsync(output, d_C, sizeof(half) * m * n, cudaMemcpyDeviceToDevice, stream));


    /* free resources */
    CHECK_CUDA( cudaFree(d_C) );
    CHECK_CUDA( cudaFree(arrA) );
    CHECK_CUDA( cudaFree(arrB) );
    CHECK_CUDA( cudaFree(arrC) );
    CHECK_CUBLAS( cublasDestroy(handle) );

}

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

    //auto ttt = new CudaTime();
    //ttt->initAndStart();
    CHECK_CUDA(cudaMalloc((void **) &dA, A_size))
    CHECK_CUDA(cudaMalloc((void **) &dB, B_size))
    // padding to match mma.sp
    padCudaMemcpy2D(inputA, inputM, inputK, dA, m, k);
    padCudaMemcpy2D(inputB, inputK, inputN, dB, k, n);
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

void cusparse_gemm_csr_device(half *sp_A, half *d_B, int m, int k, int n, half *output) {
    float alpha = 1.0f, beta = 0.0f;

    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle) )

    int ldA = k, ldB = n, ldC = n;
    half *dA_csr_values, *dC;
    int *dA_csr_offsets, *dA_csr_columns;

    CHECK_CUDA( cudaMalloc((void **)&dA_csr_offsets, sizeof(int) * (m + 1)) )
    CHECK_CUDA( cudaMalloc((void **)&dC, sizeof(half) * m * n) )

    cusparseDnMatDescr_t matA, matB, matC;
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, m, k, ldA, sp_A, CUDA_R_16F, CUSPARSE_ORDER_ROW) )
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, k, n, ldB, d_B, CUDA_R_16F, CUSPARSE_ORDER_ROW) )
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, m, n, ldC, dC, CUDA_R_16F, CUSPARSE_ORDER_ROW) )

    cusparseSpMatDescr_t matA_cmpr;
    // 此时相关参数都未设置
    CHECK_CUSPARSE( cusparseCreateCsr(&matA_cmpr, m, k, 0, dA_csr_offsets,
                                      nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )

    void* dBuffer = nullptr;
    size_t bufferSize = 0;
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(handle, matA, matA_cmpr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, matA, matA_cmpr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer) )
    // analysis后可以获取相应指针
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matA_cmpr, &num_rows_tmp, &num_cols_tmp, &nnz) )

    // allocate CSR column indices and values
    CHECK_CUDA( cudaMalloc((void**) &dA_csr_columns, nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_csr_values,  nnz * sizeof(half)) )
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matA_cmpr, dA_csr_offsets, dA_csr_columns, dA_csr_values) )

    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, matA, matA_cmpr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer) )

    // calculate
    // 当 A/B/C 都为 CUDA_R_16F，computeType 需要为 CUDA_R_32F
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matA_cmpr, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    CHECK_CUSPARSE( cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                 matA_cmpr, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    CHECK_CUDA( cudaMemcpy(output, dC, sizeof(half) * m * n, cudaMemcpyDeviceToHost) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA_cmpr) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csr_offsets) )
    CHECK_CUDA( cudaFree(dA_csr_columns) )
    CHECK_CUDA( cudaFree(dA_csr_values) )
}

void cusparse_gemm_blocked_device_test() {
    int   A_num_rows      = 4;
    int   A_num_cols      = 4;
    int   A_ell_blocksize = 2;
    int   A_ell_cols      = 2;
    int   A_num_blocks    = A_ell_cols * A_num_rows /
                            (A_ell_blocksize * A_ell_blocksize);
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = 3;
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
    int   hA_columns[]    = { 1, 0};
    __half hA_values[]    = { 1.0f, 2.0f, 3.0f, 4.0f,
                              5.0f, 6.0f, 7.0f, 8.0f};
    __half hB[]           = { 1.0f,  2.0f,  3.0f,  4.0f,
                              5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f };
    __half hC[]           = { 0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f };
    __half hC_result[]    = { 11.0f, 25.0f,  17.0f,  23.0f,
                              23.0f, 53.0f,  61.0f,  83.0f,
                              35.0f, 81.0f, 105.0f, 143.0f };
    float alpha           = 1.0f;
    float beta            = 0.0f;
    //--------------------------------------------------------------------------
    // Check compute capability
    cudaDeviceProp props{};
    CHECK_CUDA( cudaGetDeviceProperties(&props, 0) )
    if (props.major < 7) {
        std::printf("cusparseSpMM with blocked ELL format is supported only "
                    "with compute capability at least 7.0\n");
        return ;
    }
    //--------------------------------------------------------------------------
    // Device memory management
    int    *dA_columns;
    __half *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_blocks * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,
                           A_ell_cols * A_num_rows * sizeof(__half)) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(__half)) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(__half)) )

    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns,
                           A_num_blocks * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                           A_ell_cols * A_num_rows * sizeof(__half),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(__half),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(__half),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in blocked ELL format
    CHECK_CUSPARSE( cusparseCreateBlockedEll(
            &matA,
            A_num_rows, A_num_cols, A_ell_blocksize,
            A_ell_cols, dA_columns, dA_values,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_16F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_16F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    auto tt = new CudaTime();
    tt->initAndStart();
    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
    printf("time: %fms\n", tt->endAndGetTime());

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(__half),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        for (int j = 0; j < B_num_cols; j++) {
            float c_value  = static_cast<float>(hC[i + j * ldc]);
            float c_result = static_cast<float>(hC_result[i + j * ldc]);
            if (c_value != c_result) {
                correct = 0; // direct floating point comparison is not reliable
                break;
            }
        }
    }

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
}
void cusparse_gemm_csr_device_test() {
    int   A_num_rows      = 4;
    int   A_num_cols      = 4;
    int   A_nnz           = 9;
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = 3;
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
    int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              6.0f, 7.0f, 8.0f, 9.0f };
    float hB[]            = { 1.0f,  2.0f,  3.0f,  4.0f,
                              5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f };
    float hC[]            = { 0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f };
    float hC_result[]     = { 19.0f,  8.0f,  51.0f,  52.0f,
                              43.0f, 24.0f, 123.0f, 120.0f,
                              67.0f, 40.0f, 195.0f, 188.0f };
    float alpha           = 1.0f;
    float beta            = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    auto tt = new CudaTime();
    tt->initAndStart();
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
    printf("time: %fms\n", tt->endAndGetTime());
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        for (int j = 0; j < B_num_cols; j++) {
            if (hC[i + j * ldc] != hC_result[i + j * ldc]) {
                correct = 0; // direct floating point comparison is not reliable
                break;
            }
        }
    }
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
}

