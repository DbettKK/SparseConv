//
// Created by dbettkk on 2022/8/2.
//
#include <cstdio>
#include "../spmma/utils/CudaTime.cuh"

const int threadsPerBlock = 512;
const int N = 20480;
const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; /* 4 */

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


int main() {
    int size = sizeof(float);

    float *hA = new float[N];
    for (int i = 0; i < N; ++i)
        hA[i] = i;

    /* 分配显存空间 */
    float *d_a;
    float *d_partial_sum;

    cudaMallocManaged((void **) &d_a, N * size);
    cudaMallocManaged((void **) &d_partial_sum, blocksPerGrid * size);

    for (int i = 0; i < N; ++i)
        d_a[i] = i;

    /* 调用内核函数 */
    auto tt = new CudaTime();
    tt->initAndStart();
    ReductionSum <<< blocksPerGrid, threadsPerBlock >>> (d_a, d_partial_sum);
    printf("gpu: %fms\n", tt->endAndGetTime());

    // cpu
    clock_t start = clock();
    float ss = 0.0;
    for (int i = 0; i < N; ++i) {
        ss += hA[i];
    }
    printf("cpu time: %fms\n", (double)(clock() - start) * 1000 / CLOCKS_PER_SEC);

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