//
// Created by dbettkk on 2022/8/1.
//

#include "MyTensor.cuh"

half *MyTensor::getTensor() const {
    return tensor;
}

void MyTensor::setTensor(half *tensor) {
    MyTensor::tensor = tensor;
}

int MyTensor::getBatch() const {
    return batch;
}

void MyTensor::setBatch(int batch) {
    MyTensor::batch = batch;
}

int MyTensor::getChannel() const {
    return channel;
}

void MyTensor::setChannel(int channel) {
    MyTensor::channel = channel;
}

int MyTensor::getWidth() const {
    return width;
}

void MyTensor::setWidth(int width) {
    MyTensor::width = width;
}

int MyTensor::getHeight() const {
    return height;
}

void MyTensor::setHeight(int height) {
    MyTensor::height = height;
}

int MyTensor::getSize() {
    return batch * channel * height * width;
}

void MyTensor::batchNorm(int out_c, MyTensor *out) {
    this->copy(out);
}

void MyTensor::relu(MyTensor *out) {
    ReLU<<<getSize() / 32 + 1, 32>>>(tensor, out->tensor, getSize());
}

void MyTensor::print_half(half *item, int batch, int channel, int width, int height) {
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < channel; j++) {
            for (int k = 0; k < width; k++) {
                for (int v = 0; v < height; v++) {
                    printf("%.2f ", __half2float(item[i * channel * width * height + j * width * height + k * height + v]));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void MyTensor::cmp_half(half *item1, half *item2, int batch, int channel, int width, int height, bool is_device) {
    int total = batch * channel * width * height;
    int diff = 0;
    if (is_device) {
        half *h1 = new half[total], *h2 = new half[total];
        cudaMemcpy(h1, item1, sizeof(half) * total, cudaMemcpyDeviceToHost);
        cudaMemcpy(h2, item2, sizeof(half) * total, cudaMemcpyDeviceToHost);
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < channel; j++) {
                for (int k = 0; k < width; k++) {
                    for (int v = 0; v < height; v++) {
                        int index = i * channel * width * height + j * width * height + k * height + v;
                        if (__half2float(h1[index]) != __half2float(h1[index])) diff++;
                    }
                }
            }
        }
        delete[] h1;
        delete[] h2;
    } else {
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < channel; j++) {
                for (int k = 0; k < width; k++) {
                    for (int v = 0; v < height; v++) {
                        int index = i * channel * width * height + j * width * height + k * height + v;
                        if (__half2float(item1[index]) != __half2float(item2[index])) diff++;
                    }
                }
            }
        }
    }
    printf("total: %d, diff: %d\n", total, diff);
}

void MyTensor::print_half_device(half *item, int batch, int channel, int width, int height) {
    int total = batch * channel * width * height;
    half *h1 = new half[total];
    cudaMemcpy(h1, item, sizeof(half) * total, cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < channel; j++) {
            for (int k = 0; k < width; k++) {
                for (int v = 0; v < height; v++) {
                    printf("%.2f ", __half2float(h1[i * channel * width * height + j * width * height + k * height + v]));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
    delete[] h1;
}

MyTensor::MyTensor(int batch, int channel, int width, int height, bool is_device) : batch(batch),
                                                                                  channel(channel), width(width),
                                                                                  height(height) {
    if (is_device) {
        cudaMalloc(&tensor, sizeof(half) * getSize());
    } else {
        tensor = new half[getSize()];
    }
}

void
MyTensor::conv2d(int conv_num, int out_channel, int kernel_w, int kernel_h, int stride, int padding, MyTensor *out) {
    MyTensor *kernel = getKernel(conv_num, out_channel, this->getChannel(), kernel_w, kernel_h);
    conv2d_device_cudnn(this->tensor, kernel->tensor, batch, channel, kernel->batch, width, height, kernel->width,
                        kernel->height, stride, padding, out->getTensor());
}

MyTensor *MyTensor::getKernel(int conv_num, int out_channel, int in_channel, int kernel_w, int kernel_h) {
    return new MyTensor(out_channel, in_channel, kernel_w, kernel_h, true, 1);
}

void MyTensor::maxpool(int kernel_size, int stride, int padding, MyTensor *out) {
    this->copy(out);
}

MyTensor* MyTensor::copyTo() {
    MyTensor *out = new MyTensor(batch, channel, width, height, true);
    cudaMemcpy(out->getTensor(), tensor, sizeof(half) * out->getSize(), cudaMemcpyDeviceToDevice);
    return out;
}

void MyTensor::copy(MyTensor *out) {
    cudaMemcpy(out->getTensor(), tensor, sizeof(half) * out->getSize(), cudaMemcpyDeviceToDevice);
}

void MyTensor::addTensor(MyTensor *add, MyTensor *out) {
    this->copy(out);
}

void MyTensor::free_tensor() {
    CHECK_CUDA(cudaFree(tensor))
}

void MyTensor::avgpool(MyTensor *out) {
    this->copy(out);
}

MyTensor::MyTensor(int batch, int channel, int width, int height, bool is_device, half init) : batch(batch),
                                                                                    channel(channel), width(width),
                                                                                    height(height) {
    if (is_device) {
        half *tmp = new half[getSize()];
        for (int i = 0; i < getSize(); i++) tmp[i] = init;
        cudaMalloc(&tensor, sizeof(half) * getSize());
        cudaMemcpy(tensor, tmp, sizeof(half) * getSize(), cudaMemcpyHostToDevice);
    } else {
        tensor = new half[getSize()];
        for (int i = 0; i < getSize(); i++) tensor[i] = init;
    }
}

void MyTensor::print(bool is_device) {
    int total = batch * channel * width * height;
    half *h1 = new half[total];
    cudaMemcpy(h1, tensor, sizeof(half) * total, cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < channel; j++) {
            for (int k = 0; k < width; k++) {
                for (int v = 0; v < height; v++) {
                    printf("%.2f ", __half2float(h1[i * channel * width * height + j * width * height + k * height + v]));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
    delete[] h1;
}

