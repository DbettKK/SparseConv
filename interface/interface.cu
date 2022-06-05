//#include"utils/util.cuh"

#include "../spmma/kernels/sparse_conv.cuh"

extern "C" {
void matmul(float *fA, float *fB, int m, int k, int n, int isValid, float *fOut);
void conv2d(float *fData, float *fKernel, const int *data_size, const int *kernel_size,
            int padding, int stride, int isValid, float *fOut, int *out_size);
}


void matmul(float *fA, float *fB, int m, int k, int n, int isValid, float *fOut) {
    half *hA = new half[m * k], *hB = new half[k * n];

    float2half_array(fA, hA, m * k);
    float2half_array(fB, hB, n * k);

    half *dA, *dB, *dOut;
    CHECK_CUDA( cudaMalloc((void **)&dA, m * k * sizeof(half)) )
    CHECK_CUDA( cudaMalloc((void **)&dB, k * n * sizeof(half)) )
    CHECK_CUDA( cudaMalloc((void **)&dOut, m * n * sizeof(half)) )
    CHECK_CUDA( cudaMemcpy(dA, hA, m * k * sizeof(half), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, k * n * sizeof(half), cudaMemcpyHostToDevice) )

    spmma_matmul(dA, dB, m, k, n, isValid, dOut, nullptr);

    half *hOut = new half[m * n];
    CHECK_CUDA( cudaMemcpy(hOut, dOut, m * n * sizeof(half), cudaMemcpyDeviceToHost) )

    half2float_array(hOut, fOut, m * n);
}

void conv2d(float *fData, float *fKernel, const int *data_size, const int *kernel_size,
            int padding, int stride, int isValid, float *fOut, int *out_size) {
    int data_n = data_size[0], data_c = data_size[1], data_h = data_size[2], data_w = data_size[3];
    int kernel_n = kernel_size[0], kernel_c = kernel_size[1], kernel_h = kernel_size[2], kernel_w = kernel_size[3];

    half *hData = new half[data_n * data_c * data_h * data_w];
    half *hKernel = new half[kernel_n * kernel_c * kernel_h * kernel_w];

    half2float_array(hData, fData, data_n * data_c * data_h * data_w);
    half2float_array(hKernel, fKernel, kernel_n * kernel_c * kernel_h * kernel_w);

    auto *data = new Tensor4d(hData, data_n, data_c, data_h, data_w);
    auto *kernel = new Tensor4d(hKernel, kernel_n, kernel_c, kernel_h, kernel_w);
    auto *param = new ConvParam(data, kernel, padding, stride, 1);

    Tensor4d *out = sparse_conv(param);

    half2float_array(out->getTensor(), fOut, out->getTotalSize());
    out_size[0] = out->getN();
    out_size[1] = out->getC();
    out_size[2] = out->getH();
    out_size[3] = out->getW();
}



