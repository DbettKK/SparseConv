//
// Created by dbettkk on 2022/8/14.
//

#include "cusparse_interface.cuh"

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
//    CHECK_CUSPARSE( cusparseCreateBlockedEll(
//            &matA,
//            A_num_rows, A_num_cols, A_ell_blocksize,
//            A_ell_cols, dA_columns, dA_values,
//            CUSPARSE_INDEX_32I,
//            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )
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
    //int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        for (int j = 0; j < B_num_cols; j++) {
            float c_value  = static_cast<float>(hC[i + j * ldc]);
            float c_result = static_cast<float>(hC_result[i + j * ldc]);
            if (c_value != c_result) {
                //correct = 0; // direct floating point comparison is not reliable
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
    //int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        for (int j = 0; j < B_num_cols; j++) {
            if (hC[i + j * ldc] != hC_result[i + j * ldc]) {
                //correct = 0; // direct floating point comparison is not reliable
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