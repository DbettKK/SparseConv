//
// Created by dbettkk on 2022/8/4.
//

#include "tests.cuh"

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
        printf("interface time: %fms\t", t1->endAndGetTime());
        t2->initAndStart();
        conv2d_device_cudnn(d_f, d_k, batch, in_c, out_c, f_w, f_h, k_w, k_h, 1, 0, d_out);
        printf("cudnn time: %fms\n", t2->endAndGetTime());
    }

    half *out = new half[out_c * batch * out_w * out_h];
    cudaMemcpy(out, d_out, sizeof(half) * out_c * batch * out_w * out_h, cudaMemcpyDeviceToHost);

    //MyTensor::print_half(out, batch, out_c, out_w, out_h);
}

void test_bn() {
    half *hf = new half[2 * 3 * 4 * 4];
    for (int i = 0; i < 96; i++) hf[i] = 2;
    half *df, *d_out;
    cudaMalloc(&df, sizeof(half) * 96);
    cudaMalloc(&d_out, sizeof(half) * 96);
    cudaMemcpy(df, hf, sizeof(half) * 96, cudaMemcpyHostToDevice);
    bn_cudnn(df, 2, 3, 4, 4, d_out);
    MyTensor::print_half_device(d_out, 2, 3, 4, 4);
}

void test_pool() {
    half *hf = new half[2 * 3 * 4 * 4];
    for (int i = 0; i < 96; i++) hf[i] = i;
    half *df, *d_out;
    cudaMalloc(&df, sizeof(half) * 96);
    cudaMalloc(&d_out, sizeof(half) * 2 * 3 * 1 * 1);
    cudaMemcpy(df, hf, sizeof(half) * 96, cudaMemcpyHostToDevice);
    pool_cudnn(df, 2, 3, 4, 4, d_out, 4, 0, 2, 3);
    MyTensor::print_half_device(d_out, 2, 3, 1, 1);
    //MyTensor::print_half_device(df, 2, 3, 4, 4);
}

void test_im2col() {
    half *hf = new half[1 * 1 * 7 * 7];
    half *hk = new half[1 * 1 * 3 * 3];
    for (int i = 0; i < 49; i++) hf[i] = i;
    half *df, *d_out;
    cudaMalloc(&df, sizeof(half) * 49);
    cudaMemcpy(df, hf, sizeof(half) * 49, cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, sizeof(half) * 25 * 9);
    im2col_cudnn(df, 1, 1, 1, 7, 7, 3, 3, 1, 0, d_out);
    half *hout = new half[25 * 9];
    cudaMemcpy(hout, d_out, sizeof(half) * 25 * 9, cudaMemcpyDeviceToHost);
    // wrong
    for (int i = 0; i < 25; i++) {
        for (int j = 0; j < 9; j++) {
            printf("%d ", __half2int_rz(hout[i * 9 + j]));
        }
        printf("\n");
    }
    printf("\n");
    // right
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 25; j++) {
            printf("%d ", __half2int_rz(hout[i * 25 + j]));
        }
        printf("\n");
    }
}