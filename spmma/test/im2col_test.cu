//
// Created by dbettkk on 2022/4/3.
//
#include <cudnn.h>

#include "../entity/ConvParam.cuh"
#include "../utils/CudaTime.cuh"

#define CHECK_CUDNN(func)                                                      \
{                                                                              \
    cudnnStatus_t status = (func);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
        printf("CUDNN failed at line %d with error: %s (%d)\n",                \
               __LINE__, cudnnGetErrorString(status), status);                 \
        return 0;                                                              \
    }                                                                          \
}

#define CHECK_CUDA_IM2COL(func)                                                      \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",          \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        return 0;                                                              \
    }                                                                          \
}

template <typename Dtype>
void test_right(const Dtype *item1, const Dtype *item2, int total) {
    int cnt = 0;
    printf("total: %d\t", total);
    for (int i = 0; i < total; i++) {
        if (typeid(item1) == typeid(half *)) {
            if (__half2float(item1[i]) != __half2float(item2[i])) {
                cnt++;
                //printf("%d : %d\n", __half2int_rz(item1[i]), __half2int_rz(item2[i]));
            }
        } else {
            if (__half2float(item1[i]) != __half2float(item2[i])) {
                cnt++;
                //printf("%f : %f\n", item1[i], item2[i]);
            }
        }

    }
    printf("diff: %d\n", cnt);
}


int func(ConvParam *param) {
    ofstream out("windows.csv", ios::app);
    int data_size = param->getData()->getTotalSize();
    int im2col_size = param->getIm2colSize();

    float *d_data, *d_im2col_cudnn, *d_im2col_caffe;
    CHECK_CUDA_IM2COL(cudaMalloc((void **)&d_data, sizeof(float) * data_size))
    CHECK_CUDA_IM2COL(cudaMalloc((void **)&d_im2col_caffe, sizeof(float) * im2col_size))
    CHECK_CUDA_IM2COL(cudaMalloc((void **)&d_im2col_cudnn, sizeof(float) * im2col_size))

    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle))
    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN( cudnnCreateTensorDescriptor(&input_descriptor) )
    CHECK_CUDNN( cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            param->getData()->getN(), param->getData()->getC(), param->getData()->getH(), param->getData()->getW()) ) // n, c, h, w

    cudnnFilterDescriptor_t kernel_descriptor;
    CHECK_CUDNN( cudnnCreateFilterDescriptor(&kernel_descriptor) )
    CHECK_CUDNN( cudnnSetFilter4dDescriptor(kernel_descriptor,  CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                            param->getKernel()->getN(), param->getKernel()->getC(), param->getKernel()->getH(), param->getKernel()->getW()) )

    cudnnConvolutionDescriptor_t conv_descriptor;
    CHECK_CUDNN( cudnnCreateConvolutionDescriptor(&conv_descriptor) )
    CHECK_CUDNN( cudnnSetConvolution2dDescriptor(conv_descriptor,
                                                 param->getPadding(), param->getPadding(), // zero-padding
                                                 param->getStride(), param->getStride(), // stride
                                                 param->getDilation(), param->getDilation(), // dilation 卷积核膨胀 膨胀后用0填充空位
            // 卷积是需要将卷积核旋转180°再进行后续的 -> CUDNN_CONVOLUTION
                                                 CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) )


    CHECK_CUDA_IM2COL( cudaMemcpy(d_data, param->getData()->getTensor(), sizeof(float) * data_size, cudaMemcpyHostToDevice) )

    auto time_cudnn = new CudaTime();
    time_cudnn->initAndStart();

    CHECK_CUDNN( cudnnIm2Col(handle, input_descriptor, d_data, kernel_descriptor, conv_descriptor, d_im2col_cudnn) )

    float cudnn_time = time_cudnn->endAndGetTime();

    auto time_caffe = new CudaTime();
    time_caffe->initAndStart();
    im2col_gpu<float>(d_data, param->getData()->getN(), param->getData()->getC(), param->getData()->getH(), param->getData()->getW(),
               param->getKernel()->getH(), param->getKernel()->getW(),
               param->getPadding(), param->getPadding(),
               param->getStride(), param->getStride(),
               param->getDilation(), param->getDilation(), d_im2col_caffe);
    float caffe_time = time_caffe->endAndGetTime();

    auto im2col_cudnn = new float[im2col_size];
    auto im2col_caffe = new float[im2col_size];

    CHECK_CUDA_IM2COL(cudaMemcpy(im2col_cudnn, d_im2col_cudnn, sizeof(float) * im2col_size, cudaMemcpyDeviceToHost))
    CHECK_CUDA_IM2COL(cudaMemcpy(im2col_caffe, d_im2col_caffe, sizeof(float) * im2col_size, cudaMemcpyDeviceToHost))
    // test_right(im2col_caffe, im2col_cudnn, im2col_size);

    out << cudnn_time << ',' << caffe_time << endl;
    //printf("cudnn: %fms, caffe: %fms\n", cudnn_time, caffe_time);

    CHECK_CUDNN( cudnnDestroy(handle))
    CHECK_CUDNN( cudnnDestroyTensorDescriptor(input_descriptor))
    CHECK_CUDNN( cudnnDestroyFilterDescriptor(kernel_descriptor))
    CHECK_CUDNN( cudnnDestroyConvolutionDescriptor(conv_descriptor))

    CHECK_CUDA_IM2COL(cudaFree(d_data))
    CHECK_CUDA_IM2COL(cudaFree(d_im2col_cudnn))
    CHECK_CUDA_IM2COL(cudaFree(d_im2col_caffe))

    delete[] im2col_cudnn;
    delete[] im2col_caffe;
    out.close();
}

void test_im2col(int dn, int dc, int dh, int dw, int kn, int kc, int kh, int kw) {
    auto data = new Tensor4d(dn, dc, dh, dw);
    auto kernel = new Tensor4d(kn, kc, kh, kw);
    data->generateRandData(50);
    kernel->generateRandData(50);
    //printf("data:\n");
    //data->print_tensor();
    auto param = new ConvParam(data, kernel, 2, 2, 2);
    func(param);

    delete data;
    delete kernel;
    delete param;
}

int main() {
    int data_param[4] = {16, 3, 256, 256};
    int kernel_param[4] = {64, 3, 7, 7};
    int data_param1[4] = {16, 3, 112, 112};
    int kernel_param1[4] = {64, 3, 5, 5};
    int data_param2[4] = {8, 3, 64, 64};
    int kernel_param2[4] = {64, 3, 5, 5};
    int data_param3[4] = {8, 3, 32, 32};
    int kernel_param3[4] = {64, 3, 3, 3};
    int data_param4[4] = {4, 3, 16, 16};
    int kernel_param4[4] = {64, 3, 3, 3};
    for (int i = 0; i < 1; i++) {
        test_im2col(data_param[0], data_param[1], data_param[2], data_param[3], kernel_param[0], kernel_param[1], kernel_param[2], kernel_param[3]);
    }
    ofstream out("windows.csv", ios::app);
    out << "===" << endl;
    out.close();
    for (int i = 0; i < 1; i++) {
        test_im2col(data_param1[0], data_param1[1], data_param1[2], data_param1[3], kernel_param1[0], kernel_param1[1], kernel_param1[2], kernel_param1[3]);
    }
    ofstream out1("windows.csv", ios::app);
    out1 << "===" << endl;
    out1.close();
    for (int i = 0; i < 1; i++) {
        test_im2col(data_param2[0], data_param2[1], data_param2[2], data_param2[3], kernel_param2[0], kernel_param2[1], kernel_param2[2], kernel_param2[3]);
    }
    ofstream out2("windows.csv", ios::app);
    out2 << "===" << endl;
    out2.close();
    for (int i = 0; i < 1; i++) {
        test_im2col(data_param3[0], data_param3[1], data_param3[2], data_param3[3], kernel_param3[0], kernel_param3[1], kernel_param3[2], kernel_param3[3]);
    }
    ofstream out3("windows.csv", ios::app);
    out3 << "===" << endl;
    out3.close();
    for (int i = 0; i < 1; i++) {
        test_im2col(data_param4[0], data_param4[1], data_param4[2], data_param4[3], kernel_param4[0], kernel_param4[1], kernel_param4[2], kernel_param4[3]);
    }
}

