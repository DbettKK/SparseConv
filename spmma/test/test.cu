//
// Created by dbettkk on 2022/3/30.
//
#include"../kernels/sparse_conv.cuh"

void test_matmul() {
    int m = 256, k = 256, n = 256;
    auto param = new MatrixParam(m, k, n);
    param->readFromBin("../data/a.bin", "../data/b.bin", "../data/c.bin");
    //param->printMatrix('A');

    half *dA, *dB;
    CHECK_CUDA( cudaMalloc((void **)&dA, m * k * sizeof(half)) )
    CHECK_CUDA( cudaMalloc((void **)&dB, k * n * sizeof(half)) )
    param->copyToDevice(dA, 'A');
    param->copyToDevice(dB, 'B');

    auto *out = new MatrixParam(m, k, n);
    half *outputD;
    CHECK_CUDA( cudaMalloc((void **)&outputD, m * n * sizeof(half)) )

    spmma_matmul(dA, dB, m, k, n, true, outputD, out);

    out->checkCorrect(false);

    CHECK_CUDA(cudaFree(dA))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(outputD))

    delete param;
}

void benchmark_matmul(int m, int k, int n) {
    auto param = new MatrixParam(m, k, n);
    param->generateRandData(100);

    half *dA, *dB;
    CHECK_CUDA( cudaMalloc((void **)&dA, m * k * sizeof(half)) )
    CHECK_CUDA( cudaMalloc((void **)&dB, k * n * sizeof(half)) )
    param->copyToDevice(dA, 'A');
    param->copyToDevice(dB, 'B');

    auto *out = new MatrixParam(m, k, n);
    half *outputD;
    CHECK_CUDA( cudaMalloc((void **)&outputD, m * n * sizeof(half)) )

    spmma_matmul(dA, dB, m, k, n, true, outputD, out);

    out->checkCorrect(false);

    CHECK_CUDA(cudaFree(dA))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(outputD))
    delete param;
}

void test_conv() {
    int data_n = 4, data_c = 3, data_h = 16, data_w = 16;
    int kernel_n = 4, kernel_c = 3, kernel_h = 3, kernel_w = 3;
    auto data = new Tensor4d(data_n, data_c, data_h, data_w);
    auto kernel = new Tensor4d(kernel_n, kernel_c, kernel_h, kernel_w);
//    data->generateRandData(8);
//    kernel->generateRandSpData(8);
    data->readFromBin("../data/data.bin");
    kernel->readFromBin("../data/kernel.bin");
    //data->printTensor("data:");
    auto param = new ConvParam(data, kernel, 0, 1, 1);
    // test1 cpu占用会很高
    if (!param->checkIm2col()) {
        printf("im2col error\n");
        return;
    }

    Tensor4d *out = sparse_conv(param);

    delete data;
    delete kernel;
    delete out;
}

void benchmark_conv(int data_n, int data_c, int data_h, int data_w, int kernel_n, int kernel_c, int kernel_h, int kernel_w) {
    auto data = new Tensor4d(data_n, data_c, data_h, data_w);
    auto kernel = new Tensor4d(kernel_n, kernel_c, kernel_h, kernel_w);
    data->generateRandData(50);
    kernel->generateRandSpData(50);
//    data->printTensor("data:");
//    kernel->printTensor("kernel:");

    auto param = new ConvParam(data, kernel, 0, 1, 1);

    // test1
    if (!param->checkIm2col()) {
        printf("im2col error\n");
        return;
    }

    Tensor4d *out = sparse_conv(param);

    delete data;
    delete kernel;
    delete out;
}

//int main() {
//    std::random_device sd; // sd可以产生一个质量很高的随机数
//    std::default_random_engine e(sd());
//    std::uniform_int_distribution<unsigned> u(1, 255); // 闭区间
//
//    int data_n = 1, data_c = 1;
//    int kernel_n = 16, kernel_c = data_c;
//    int data_h = 7, data_w = 7;
//    int kernel_h = 4;
//    int kernel_w = kernel_h;
//
//    printf("DATA SIZE: %d,%d,%d,%d\n", data_n, data_c, data_h, data_w);
//    printf("KERNEL SIZE: %d,%d,%d,%d\n", kernel_n, kernel_c, kernel_h, kernel_w);
//    printf("padding: %d, stride: %d, dilation: %d\n", 0, 1, 1);
//
//    for(int i = 0; i < 20; i++) benchmark_conv(data_n, data_c, data_h, data_w, kernel_n, kernel_c, kernel_h, kernel_w);
//
//}

//int main() {
    // benchmark matmul
//    std::random_device sd; // sd可以产生一个质量很高的随机数
//    std::default_random_engine e(sd());
//    std::uniform_int_distribution<unsigned> u(16, 200); // 闭区间
//    for (int i = 0; i < 20; i++) {
//        int m = u(e), k = u(e), n = u(e);
//        printf("m: %d,k: %d,n: %d\t", m, k, n);
//        benchmark_matmul(m, k, n);
//    }
    //for (int i = 0; i < 1; i++) test_conv();
//    for (int i = 0; i < 50; i++) {
//        std::random_device sd; // sd可以产生一个质量很高的随机数
//        std::default_random_engine e(sd());
//        std::uniform_int_distribution<unsigned> u_data_wh(16, 50); // 闭区间
//        std::uniform_int_distribution<unsigned> u_kernel_wh(3, 16); // 闭区间
//        int data_n = 1;
//        int data_c = 1;
//        int data_h = u_data_wh(e);
//        int data_w = data_h;
//        int kernel_n = 1;
//        int kernel_c = data_c;
//        int kernel_h = u_kernel_wh(e);
//        int kernel_w = kernel_h;
//        benchmark_conv(data_n, data_c, data_h, data_w, kernel_n, kernel_c, kernel_h, kernel_w);
//    }
//    fflush(stdout);
    //for (int i = 0; i < 50; i++) test_conv();
    //test_conv();
    // todo: 只compress的截图
    //return 0;
//}

