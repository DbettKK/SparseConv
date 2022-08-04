//
// Created by dbettkk on 2022/8/2.
//
#include <cstdio>
#include "../spmma/utils/CudaTime.cuh"

const int threadsPerBlock = 512;
const int N = 512 * 16;
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

__global__ void layerSum(float *d, float *sum) {
    int thx = threadIdx.x;
    float sums = 0;
    for (int i = 0; i < 512; i++) sums += d[thx * 512 + i];
    //ReductionSum <<< 1, threadsPerBlock >>> (d, sum + thx);
    sum[thx] = sums;
}
int main() {
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