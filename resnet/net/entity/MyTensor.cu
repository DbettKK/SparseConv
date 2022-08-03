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
    /*auto bn_t = new CudaTime();
    bn_t->initAndStart();
    half *h_tensor = new half[getSize()];
    cudaMemcpy(h_tensor, tensor, sizeof(half) * getSize(), cudaMemcpyDeviceToHost);
    auto mean = new float[channel];
    //auto mean_2 = new float[channel];
    auto std = new float[channel];
    for (int i = 0; i < channel; i++) {
        float m1 = 0.0, m2 = 0.0;
        for (int b = 0; b < batch; b++) {
            for (int w = 0; w < width; w++) {
                for (int h = 0; h < height; h++) {
                    int idx = b * channel * width * height + i * width * height + w * height + h;
                    float item = __half2float(h_tensor[idx]);
                    m1 += item / (float) (batch * width * height);
                    m2 += item / (float) (batch * width * height) * item;
                }
            }
        }
        mean[i] = m1;
        std[i] = sqrt(m2 - m1 * m1);
    }
    // cpu
    for (int i = 0; i < channel; i++) {
        for (int b = 0; b < batch; b++) {
            for (int w = 0; w < width; w++) {
                for (int h = 0; h < height; h++) {
                    int idx = b * channel * width * height + i * width * height + w * height + h;
                    float item = __half2float(h_tensor[idx]);
                    h_tensor[idx] = (item - mean[i]) / std[i];
                }
            }
        }
    }
    cudaMemcpy(out->getTensor(), h_tensor, sizeof(half) * getSize(), cudaMemcpyHostToDevice);*/
    //printf("bn time: %fms\n", bn_t->endAndGetTime());
    //auto bn_t = new CudaTime();
    //bn_t->initAndStart();
    bn_cudnn(tensor, batch, channel, width, height, out->getTensor());
    //printf("bn time: %fms\n", bn_t->endAndGetTime());
    //this->copy(out);
}

void MyTensor::relu(MyTensor *out) {
    //auto relu_t = new CudaTime();
    //relu_t->initAndStart();
    ReLU<<<getSize() / 32 + 1, 32>>>(tensor, out->tensor, getSize());
    //printf("relu time: %fms\n", relu_t->endAndGetTime());
}

void MyTensor::print_half(half *item, int batch, int channel, int width, int height) {
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < channel; j++) {
            for (int k = 0; k < width; k++) {
                for (int v = 0; v < height; v++) {
                    printf("%.2f ",
                           __half2float(item[i * channel * width * height + j * width * height + k * height + v]));
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
                    printf("%.2f ",
                           __half2float(h1[i * channel * width * height + j * width * height + k * height + v]));
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
    std::string path = "../../data/resnet50/filter" + std::to_string(conv_num);
    auto kernel = new MyTensor(out_channel, this->getChannel(), kernel_w, kernel_h, true, path);
    //kernel->print(true);
    conv2d_device_cudnn(this->tensor, kernel->tensor, batch, channel, kernel->batch, width, height, kernel->width,
                        kernel->height, stride, padding, out->getTensor());
    //auto conv_t = new CudaTime();
    //conv_t->initAndStart();
    //printf("conv%d\n", conv_num);
    //conv2d_device_spmma(this->tensor, kernel->tensor, batch, channel, kernel->batch, width, height, kernel->width,
    //                    kernel->height, stride, padding, out->getTensor());
    //printf("conv%d time: %fms ", conv_num, conv_t->endAndGetTime());
    kernel->free_tensor();
}

MyTensor *MyTensor::getKernel(int conv_num, int out_channel, int in_channel, int kernel_w, int kernel_h) {
    std::string path = "../../data/resnet50/filter";
    std::ifstream in(path + std::to_string(conv_num), std::ios::binary);

    auto ret = new MyTensor(out_channel, in_channel, kernel_w, kernel_h, true);
    half *kernel = new half[ret->getSize()];
    in.read((char *)kernel, ret->getSize() * sizeof(half));
    cudaMemcpy(ret->getTensor(), kernel, sizeof(half) * ret->getSize(), cudaMemcpyHostToDevice);
    in.close();

    return ret;
}

void MyTensor::maxpool(int kernel_size, int stride, int padding, MyTensor *out) {
    pool_cudnn(tensor, batch, channel, width, height, out->getTensor(), kernel_size, padding, stride, 1);
}

MyTensor *MyTensor::copyTo() {
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
    pool_cudnn(tensor, batch, channel, width, height, out->getTensor(), width, 0, 2, 0);
    //auto avg_t = new CudaTime();
    //avg_t->initAndStart();
    /*half *h_item = new half[this->getSize()];
    half *h_out = new half[batch * channel];
    int cnt = 0;
    cudaMemcpy(h_item, tensor, sizeof(half) * getSize(), cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < channel; j++) {
            float mean = 0.0;
            for (int w = 0; w < width; w++) {
                for (int h = 0; h < height; h++) {
                    int idx = i * channel * height * width + j * width * height + w * height + h;
                    mean += __half2float(h_item[idx]);
                }
            }
            h_out[cnt++] = mean / (float)(width * height);
        }
    }
    cudaMemcpy(out->getTensor(), h_out, sizeof(half) * batch * channel, cudaMemcpyHostToDevice);
    delete[] h_out;
    delete[] h_item;
    //this->copy(out);
    //printf("avg time: %fms\n", avg_t->endAndGetTime());*/
}

MyTensor::MyTensor(int batch, int channel, int width, int height, bool is_device, half init) : batch(batch),
                                                                                               channel(channel),
                                                                                               width(width),
                                                                                               height(height) {
    int tt = batch * channel * width * height;
    if (is_device) {
        half *tmp = new half[tt];
        for (int i = 0; i < tt; i++) tmp[i] = init;
        cudaMalloc(&tensor, sizeof(half) * tt);
        cudaMemcpy(tensor, tmp, sizeof(half) * tt, cudaMemcpyHostToDevice);
    } else {
        tensor = new half[tt];
        for (int i = 0; i < tt; i++) tensor[i] = init;
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
                    printf("%.3f ",
                           __half2float(h1[i * channel * width * height + j * width * height + k * height + v]));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
    delete[] h1;
}

MyTensor::MyTensor(int batch, int channel, int width, int height, bool is_device, std::string path) : batch(batch),
                                                                                                       channel(channel),
                                                                                                       width(width),
                                                                                                       height(height) {
    std::ifstream in(path, std::ios::binary);
    if (is_device) {
        half *kernel = new half[getSize()];
        in.read((char *)kernel, getSize() * sizeof(half));
        CHECK_CUDA(cudaMalloc(&tensor, sizeof(half) * getSize()))
        CHECK_CUDA(cudaMemcpy(tensor, kernel, sizeof(half) * getSize(), cudaMemcpyHostToDevice))
        in.close();
    } else {
        in.read((char *)tensor, getSize() * sizeof(half));
        in.close();
    }
}

