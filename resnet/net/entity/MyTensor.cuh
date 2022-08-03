//
// Created by dbettkk on 2022/8/1.
//

#ifndef SPARSECONV_MYTENSOR_CUH
#define SPARSECONV_MYTENSOR_CUH

#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <string>
#include "../../interface/interfaces.cuh"

class MyTensor {
    half *tensor{};
    int batch, channel, width, height;

public:
    MyTensor(int batch, int channel, int width, int height, bool is_device);

    MyTensor(int batch, int channel, int width, int height, bool is_device, half init);

    MyTensor(int batch, int channel, int width, int height, bool is_device, std::string path);

    MyTensor* copyTo();

    void copy(MyTensor *out);

    static void print_half(half *item, int batch, int channel, int width, int height);

    static void print_half_device(half *item, int batch, int channel, int width, int height);

    void print(bool is_device);

    static void cmp_half(half *item1, half *item2, int batch, int channel, int width, int height, bool is_device);

    half *getTensor() const;

    void setTensor(half *tensor);

    int getBatch() const;

    void setBatch(int batch);

    int getChannel() const;

    void setChannel(int channel);

    int getWidth() const;

    void setWidth(int width);

    int getHeight() const;

    void setHeight(int height);

    int getSize();

    void batchNorm(int out_c, MyTensor *out);

    void relu(MyTensor *out);

    static MyTensor *getKernel(int conv_num, int out_channel, int in_channel, int kernel_w, int kernel_h);

    void maxpool(int kernel_size, int stride, int padding, MyTensor *out);

    void avgpool(MyTensor *out);

    void addTensor(MyTensor *add, MyTensor *out);

    void conv2d(int conv_num, int out_channel, int kernel_w, int kernel_h, int stride, int padding, MyTensor *out);

    void free_tensor();
};


#endif //SPARSECONV_MYTENSOR_CUH
