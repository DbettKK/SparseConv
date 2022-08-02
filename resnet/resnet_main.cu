//
// Created by dbettkk on 2022/8/1.
//

#include"interface/interfaces.cuh"
#include "net/entity/MyTensor.cuh"
#include "net/MyResNet.cuh"

void test_conv2d() {
    int batch = 4, in_c = 3, f_w = 16, f_h = 16;
    int out_c = 64, k_w = 4, k_h = 4;
    half *feature = new half[batch * in_c * f_w * f_h];
    half *kernel = new half[out_c * in_c * k_w * k_h];
    for (int i = 0; i < batch * in_c * f_w * f_h; i++) feature[i] = 1;
    for (int i = 0; i < out_c * in_c * k_w * k_h; i+=2) {
        kernel[i] = 1;
        kernel[i + 1] = 0;
    }
    half *d_f, *d_k;
    cudaMalloc(&d_f, sizeof(half) * batch * in_c * f_w * f_h);
    cudaMalloc(&d_k, sizeof(half) * out_c * in_c * k_w * k_h);
    cudaMemcpy(d_f, feature, sizeof(half) * batch * in_c * f_w * f_h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, kernel, sizeof(half) * out_c * in_c * k_w * k_h, cudaMemcpyHostToDevice);

    half *d_out;
    int out_w = (f_w + 2 * 0 - k_w) / 1 + 1;
    int out_h = (f_h + 2 * 0 - k_h) / 1 + 1;
    cudaMalloc(&d_out, sizeof(half) * out_c * batch * out_w * out_h);
    for (int i = 0; i < 12; i++) {
        auto t1 = new CudaTime();
        auto t2 = new CudaTime();
        t1->initAndStart();
        conv2d_device_spmma(d_f, d_k, batch, in_c, out_c, f_w, f_h, k_w, k_h, 1, 0, d_out);
        printf("spmma time: %fms\t", t1->endAndGetTime());
        t2->initAndStart();
        conv2d_device_cudnn(d_f, d_k, batch, in_c, out_c, f_w, f_h, k_w, k_h, 1, 0, d_out);
        printf("cudnn time: %fms\n", t2->endAndGetTime());
    }

    half *out = new half[out_c * batch * out_w * out_h];
    cudaMemcpy(out, d_out, sizeof(half) * out_c * batch * out_w * out_h, cudaMemcpyDeviceToHost);

    //MyTensor::print_half(out, batch, out_c, out_w, out_h);
}


void test_resnet() {
    auto mr = new MyResNet();
    mr->resnet50();
}

void test_bn() {
    MyTensor *t = new MyTensor(2, 2, 4, 4, false, 1);
    for (int i = 0; i < 64; i++) t->getTensor()[i] = i;
    half *dd;
    cudaMalloc(&dd, sizeof(half) * 64);
    cudaMemcpy(dd, t->getTensor(), sizeof(half) * 64, cudaMemcpyHostToDevice);
    t->setTensor(dd);
    t->batchNorm(1, nullptr);
}

int main() {
    test_bn();
    return 0;
}