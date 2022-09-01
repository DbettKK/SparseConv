//
// Created by dbettkk on 2022/8/2.
//
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cusparseLt.h>
#include "../spmma/utils/CudaTime.cuh"

const int threadsPerBlock = 512;
const int N = 512 * 16;
const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; /* 4 */

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",          \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        return;                                                                \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at %s line %d with error:  (%d)\n",          \
           __FILE__, __LINE__, status);                                        \
        return ;                                                               \
    }                                                                          \
}
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at %s line %d with error: %s (%d)\n",      \
               __FILE__, __LINE__, cusparseGetErrorString(status), status);    \
        return;                                                                \
    }                                                                          \
}
__global__ void ReductionSum(float *d_a, float *d_partial_sum) {
    /* 申请共享内存, 存在于每个block中 */
    __shared__ float partialSum[threadsPerBlock];

    /* 确定索引 */
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    /* 传global memory数据到shared memory */
    partialSum[tid] = d_a[i];

    /* 传输同步 */
    __syncthreads();

    /* 在共享存储器中进行规约 */
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)
            partialSum[tid] += partialSum[tid + stride];
        __syncthreads();
    }

    /* 将当前block的计算结果写回输出数组 */
    if (tid == 0)
        d_partial_sum[blockIdx.x] = partialSum[0];
}

__global__ void layerSum(float *d, float *sum) {
    int thx = threadIdx.x;
    float sums = 0;
    for (int i = 0; i < 512; i++) sums += d[thx * 512 + i];
    //ReductionSum <<< 1, threadsPerBlock >>> (d, sum + thx);
    sum[thx] = sums;
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
            //printf("!!!! The matrix need to be pruned. valid: %d\n", *is_valid);
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
    //dim3 grid(m / 32 + 1, n / 32 + 1);
    //dim3 block(32, 32);
    //transpose_half<<<grid, block>>>(d_C, output, m, n);

    /* step 4: copy data to host */
    CHECK_CUDA( cudaMemcpy(output, d_C, sizeof(half) * m * n, cudaMemcpyDeviceToDevice));

    /* free resources */
    CHECK_CUDA( cudaFree(d_C) );
    CHECK_CUBLAS( cublasDestroy(cublasH) );

}

void cublas_gemm_batches_device(half *d_A, half *d_B, int batch, int inputM, int inputK, int inputN, half *output) {
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
//    half **dArrA, **dArrB, **dArrC;
//    half *arrA[64], *arrB[64], *arrC[64];
//    CHECK_CUDA(cudaMalloc(&dArrA, sizeof(half*) * batch))
//    CHECK_CUDA(cudaMalloc(&dArrB, sizeof(half*) * batch))
//    CHECK_CUDA(cudaMalloc(&dArrC, sizeof(half*) * batch))
//    for (int i = 0; i < batch; i++) {
//        CHECK_CUDA(cudaMalloc(&arrA[i], sizeof(half) * m * k))
//        CHECK_CUDA(cudaMalloc(&arrB[i], sizeof(half) * n * k))
//        CHECK_CUDA(cudaMalloc(&arrC[i], sizeof(half) * m * n))
//        CHECK_CUDA(cudaMemcpy(&dArrA[i], &arrA[i], sizeof(half*), cudaMemcpyHostToDevice))
//        CHECK_CUDA(cudaMemcpy(&dArrB[i], &arrB[i], sizeof(half*), cudaMemcpyHostToDevice))
//        CHECK_CUDA(cudaMemcpy(&dArrC[i], &arrC[i], sizeof(half*), cudaMemcpyHostToDevice))
//        CHECK_CUDA(cudaMemcpy(arrA[i], d_A + i * m * k, sizeof(half) * m * k, cudaMemcpyDeviceToDevice))
//        CHECK_CUDA(cudaMemcpy(arrB[i], d_B + i * n * k, sizeof(half) * k * n, cudaMemcpyDeviceToDevice))
//        CHECK_CUDA(cudaMemcpy(arrC[i], d_C + i * m * n, sizeof(half) * m * n, cudaMemcpyDeviceToDevice))
//    }
    half **dArrA, **dArrB, **dArrC;
    half *arrA[64], *arrB[64], *arrC[64];
    for (int i = 0; i < batch; i++) {
        half *tmpA, *tmpB, *tmpC;
        CHECK_CUDA(cudaMalloc(&tmpA, sizeof(half) * m * k))
        CHECK_CUDA(cudaMalloc(&tmpB, sizeof(half) * n * k))
        CHECK_CUDA(cudaMalloc(&tmpC, sizeof(half) * m * n))
        CHECK_CUDA(cudaMemcpy(tmpA, d_A + i * m * k, sizeof(half) * m * k, cudaMemcpyDeviceToDevice))
        CHECK_CUDA(cudaMemcpy(tmpB, d_B, sizeof(half) * n * k, cudaMemcpyDeviceToDevice))
        CHECK_CUDA(cudaMemcpy(tmpC, d_C + i * m * n, sizeof(half) * m * n, cudaMemcpyDeviceToDevice))
        arrA[i] = tmpA;
        arrB[i] = tmpB;
        arrC[i] = tmpC;
//        arrA[i] = d_A + i * m * k;
//        arrB[i] = d_B + i * n * k;
//        arrC[i] = d_C + i * m * n;
    }
    CHECK_CUDA(cudaMalloc(&dArrA, sizeof(half*) * batch))
    CHECK_CUDA(cudaMalloc(&dArrB, sizeof(half*) * batch))
    CHECK_CUDA(cudaMalloc(&dArrC, sizeof(half*) * batch))
    CHECK_CUDA(cudaMemcpy(dArrA, arrA, sizeof(half*) * batch, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dArrB, arrB, sizeof(half*) * batch, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dArrC, arrC, sizeof(half*) * batch, cudaMemcpyHostToDevice))

    CHECK_CUBLAS(cublasHgemmBatched(handle, transa, transb, m, n, k, &alpha, dArrA, lda, dArrB, ldb, &beta, dArrC, ldc, batch))

    CHECK_CUDA(cudaMemcpy(arrC, dArrC, sizeof(half*) * batch, cudaMemcpyDeviceToHost))


    half *c_out;
    CHECK_CUDA(cudaMalloc(&c_out, sizeof(half) * batch * m * n))
    for (int i = 0; i < batch; i++) {
        CHECK_CUDA(cudaMemcpy(output + i * m * n, arrC[i], sizeof(half) * m * n, cudaMemcpyDeviceToDevice))
        //cudaMemcpy(c_out, dArrC + i, sizeof(half) * m * n, cudaMemcpyDeviceToHost);
//        for (int j = 0; j < m; j++) {
//            for (int v = 0; v < n; v++) {
//                printf("%d ", __half2int_rz(c_out[j * n + v]));
//            }
//            printf("\n");
//        }
    }

    // transpose
//    dim3 grid(m / 32 + 1, n / 32 + 1);
//    dim3 block(32, 32);
//    transpose_half<<<grid, block>>>(d_C, output, m, n);

    /* step 4: copy data to host */
    //CHECK_CUDA( cudaMemcpyAsync(output, d_C, sizeof(half) * m * n, cudaMemcpyDeviceToDevice, stream));


    /* free resources */
    CHECK_CUDA( cudaFree(d_C) );
    //CHECK_CUDA( cudaFree(arrA) );
    //CHECK_CUDA( cudaFree(arrB) );
    //CHECK_CUDA( cudaFree(arrC) );
    CHECK_CUBLAS( cublasDestroy(handle) );

}

void test_blas() {
    half *h_A = new half[16 * 256 * 256];
    half *h_B = new half[16 * 256 * 256];
    half *h_out = new half[16 * 16 * 256];
    for (int i = 0; i < 16 * 256 * 256; i++) h_A[i] = 3;
    for (int i = 0; i < 16 * 256 * 256; i++) h_B[i] = 4;

    half *dA, *dB, *dOut;
    cudaMalloc(&dA, sizeof(half) * 16 * 256 * 256);
    cudaMalloc(&dB, sizeof(half) * 256 * 256 * 16);
    cudaMalloc(&dOut, sizeof(half) * 16 * 256 * 256);
    cudaMemcpy(dA, h_A, sizeof(half)*16 * 256 * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, h_B, sizeof(half)*16 * 256 * 256, cudaMemcpyHostToDevice);

    for (int i = 0; i < 1; i++) {
        auto tt0 = new CudaTime();
        tt0->initAndStart();
        for (int j = 0; j < 16; j++) {
            cublas_gemm_device(dA + j * 256 * 256, dB + j * 256 * 256, 256, 256, 256, dOut + j * 256 * 256);
        }
        printf("no batch time: %fms\n", tt0->endAndGetTime());
    }
    for (int i = 0; i < 1; i++) {
        auto tt0 = new CudaTime();
        tt0->initAndStart();
        for (int j = 0; j < 16; j++) {
            sparse_mma_gemm_device(dA + j * 256 * 256, dB + j * 256 * 256, 256, 256, 256, false, dOut + j * 256 * 256);
        }
        printf("no batch interface time: %fms\n", tt0->endAndGetTime());
    }
    for (int i = 0; i < 1; i++) {
        auto tt1 = new CudaTime();
        tt1->initAndStart();
        cublas_gemm_batches_device(dA, dB, 16, 256, 256, 256, dOut);
        printf("batch time: %fms\n", tt1->endAndGetTime());
    }

}

int main2() {
    test_blas();
    return 0;
    int size = sizeof(float);

    float *hA = new float[N];
    for (int i = 0; i < N; ++i)
        hA[i] = i;

    /* 分配显存空间 */
    float *d_a;
    float *d_partial_sum;
    float *d_sum;

    cudaMallocManaged((void **) &d_a, N * size);
    cudaMallocManaged((void **) &d_partial_sum, blocksPerGrid * size);
    cudaMalloc(&d_sum, 16 * sizeof(float));

    for (int i = 0; i < N; ++i)
        d_a[i] = i;

    /* 调用内核函数 */
    auto tt = new CudaTime();
    tt->initAndStart();
    layerSum<<<1, 16>>>(d_a, d_sum);
    printf("gpu: %fms\n", tt->endAndGetTime());

    auto tt3 = new CudaTime();
    tt3->initAndStart();
    ReductionSum <<< blocksPerGrid, threadsPerBlock >>> (d_a, d_partial_sum);
    printf("gpu2: %fms\n", tt3->endAndGetTime());

    // cpu
    auto tt2 = new CudaTime();
    tt2->initAndStart();
    float *ss = new float[16];
    for (int j = 0; j < 16; j++) {
        float sss = 0.0;
        for (int i = 0; i < N; ++i) {
            sss += hA[j * N + i];
        }
        ss[j] = sss;
    }
    printf("cpu time: %fms\n", tt2->endAndGetTime());

    /* 将部分和求和 */
    int sum = 0;
    for (int i = 0; i < blocksPerGrid; ++i)
        sum += d_partial_sum[i];

    printf("sum = %d\n", sum);

    /* 释放显存空间 */
    cudaFree(d_a);
    cudaFree(d_partial_sum);

    return (0);
}