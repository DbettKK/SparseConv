//
// Created by dbettkk on 2022/3/30.
//

#include "CudaTime.cuh"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",          \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        return;                                                                \
    }                                                                          \
}

#define CHECK_CUDA_NO_RET(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",          \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        return 0;                                                              \
    }                                                                          \
}

void CudaTime::init() {
    CHECK_CUDA( cudaEventCreateWithFlags(&startTime, cudaEventBlockingSync) )
    CHECK_CUDA( cudaEventCreateWithFlags(&endTime, cudaEventBlockingSync) )
    CHECK_CUDA( cudaEventCreate(&startTime) )
    CHECK_CUDA( cudaEventCreate(&endTime) )
}

void CudaTime::start() {
    CHECK_CUDA( cudaEventRecord(startTime) )
}

void CudaTime::end() {
    CHECK_CUDA( cudaEventRecord(endTime) )
}

float CudaTime::getTime() {
    float totalTime;
    //cudaEventQuery(endTime);
    CHECK_CUDA_NO_RET( cudaEventSynchronize(endTime) )
    CHECK_CUDA_NO_RET( cudaEventElapsedTime(&totalTime, startTime, endTime) )
    return totalTime;
}

void CudaTime::destroy() {
    CHECK_CUDA( cudaEventDestroy(startTime) )
    CHECK_CUDA( cudaEventDestroy(endTime) )
}

void CudaTime::initAndStart() {
    init();
    start();
}

float CudaTime::endAndGetTime() {
    end();
    float totalTime = getTime();
    destroy();
    return totalTime;
}

void CudaTime::endAndPrintTime(const std::string &msg) {
    printf("%s %fms\n", msg.c_str(), endAndGetTime());
}

void CudaTime::endAndExportTimeToFile(const std::string path, const std::string prefix_msg) {
    float time = endAndGetTime();
    std::ofstream out(path, std::ios::app);
    out << prefix_msg << time << "ms\n";
    out.close();
}
