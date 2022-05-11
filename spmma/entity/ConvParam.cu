//
// Created by dbettkk on 2022/3/29.
//

#include "ConvParam.cuh"

ConvParam::ConvParam() {}

ConvParam::ConvParam(Tensor4d *data, Tensor4d *kernel, int padding, int stride, int dilation) : data(data),
                                                                                                kernel(kernel),
                                                                                                padding(padding),
                                                                                                stride(stride),
                                                                                                dilation(dilation) {}

int ConvParam::getOutHeight() const {
    return (data->getH() + 2 * padding - (dilation * (kernel->getH() - 1) + 1)) / stride + 1;
}

int ConvParam::getOutWidth() const {
    return (data->getW() + 2 * padding - (dilation * (kernel->getW() - 1) + 1)) / stride + 1;
}

int ConvParam::getIm2colSize() const {
    return data->getN() * getOutHeight() * getOutWidth() * kernel->getC() * kernel->getH() * kernel->getW();
}

int ConvParam::getM() const {
    return data->getN() * getOutHeight() * getOutWidth();
}

int ConvParam::getK() const {
    return kernel->getC() * kernel->getH() * kernel->getW();
}

 int ConvParam::getN() const {
    return kernel->getN();
}

Tensor4d *ConvParam::padData() {
    int data_h_pad = data->getH() + padding * 2;
    int data_w_pad = data->getW() + padding * 2;

    auto data_pad = new float[data->getN() * data->getC() * data_h_pad * data_w_pad];

    for (int i = 0; i < data->getN(); i++) {
        for (int j = 0; j < data->getC(); j++) {
            int index1 = i * data->getC() * data_h_pad * data_w_pad + j * data_h_pad * data_w_pad;
            for (int ki = 0; ki < padding; ki++) {
                for (int v = 0; v < data_w_pad; v++) {
                    data_pad[index1 + ki * data_w_pad + v] = 0;
                }
            }
            for (int ki = padding; ki < padding + data->getH(); ki++) {
                for (int v = 0; v < data_w_pad; v++) {
                    if (v < padding || v >= data->getW() + padding) data_pad[index1 + ki * data_w_pad + v] = 0;
                    else data_pad[index1 + ki * data_w_pad + v] = data->getTensor()[i * data->getC() * data->getH() * data->getW() + j * data->getH() * data->getW() + (ki - padding) * data->getW() + v - padding];
                }
            }
            for (int ki = data_h_pad - padding; ki < data_h_pad; ki++) {
                for (int v = 0; v < data_w_pad; v++) {
                    data_pad[index1 + ki * data_w_pad + v] = 0;
                }
            }
        }
    }
    return new Tensor4d(data_pad, data->getN(), data->getC(), data_h_pad, data_w_pad);
}

float *ConvParam::data2col() {
    // 不支持dilation
    // m = data_n * out_w * out_h
    // k = kernel_c * kernel_w * kernel_h
    // kernel_c == data_c
    int out_h = getOutHeight(), out_w = getOutWidth();

    auto ret = new float[getIm2colSize()];

    // padding
    Tensor4d *data_pad = padData();

    // im2col4d
    int cnt = 0;
    for (int ni = 0; ni < data_pad->getN(); ni++) {
        int index_n = ni * data_pad->getC() * data_pad->getH() * data_pad->getW();
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                for (int ci = 0; ci < data_pad->getC(); ci++) {
                    int index_c = ci * data_pad->getH() * data_pad->getW();
                    int row_num = i * stride, col_num = j * stride;
                    for (int ki = row_num; ki < row_num + kernel->getH(); ki++) {
                        for (int v = col_num; v < col_num + kernel->getW(); v++) {
                            if (ki >= data_pad->getH() || v >= data_pad->getW())
                                ret[cnt++] = 0;
                            else
                                ret[cnt++] = data_pad->getTensor()[index_n + index_c + ki * data_pad->getW() + v];
                        }
                    }
                }
            }
        }
    }

    delete data_pad;
    return ret;
}

float *ConvParam::kernel2col() {
    auto ret = new float[kernel->getTotalSize()];
    auto tmp = new float[kernel->getTotalSize()];

    int k = getK();
    int cnt = 0;
    for (int i = 0; i < kernel->getN(); i++) {
        for (int j = 0; j < kernel->getC(); j++) {
            for (int ki = 0; ki < kernel->getH(); ki++) {
                for (int v = 0; v < kernel->getW(); v++) {
                    int index = i * kernel->getC() * kernel->getH() * kernel->getW() + j * kernel->getH() * kernel->getW() + ki * kernel->getW() + v;
                    tmp[cnt++] = kernel->getTensor()[index];
                }
            }
        }
    }

    // transpose
    int retCnt = 0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < kernel->getN(); j++) {
            ret[retCnt++] = tmp[j * k + i];
        }
    }

    delete[] tmp;
    return ret;
}

MatrixParam *ConvParam::im2col()  {
    int m = getM();
    int k = getK();
    int n = getN();

    float *A = data2col();
    float *B = kernel2col();

    auto C = new float[m * n];
    auto D = new float[m * n];
    memset(C, 0, m * n * sizeof(float));
    memset(D, 0, m * n * sizeof(float));

    return new MatrixParam(m, k, n, A, B, C, D);
}

Tensor4d *ConvParam::col2im(MatrixParam *param) {
    int out_h = getOutHeight(), out_w = getOutWidth();
    float *ans = param->getMatD();
    auto ret = new float[getM() * getN()];

    int cnt = 0;
    for (int i = 0; i < data->getN(); i++) {
        for (int j = 0; j < kernel->getN(); j++) {
            for (int v = 0; v < out_h * out_w; v++) {
                ret[cnt++] = ans[(i * out_h * out_w + v) * kernel->getN() + j];
            }
        }
    }

    return new Tensor4d(ret, data->getN(), kernel->getN(), out_h, out_w);
}

Tensor4d *ConvParam::getData() const {
    return data;
}

Tensor4d *ConvParam::getKernel() const {
    return kernel;
}

int ConvParam::getPadding() const {
    return padding;
}

int ConvParam::getStride() const {
    return stride;
}

int ConvParam::getDilation() const {
    return dilation;
}

void ConvParam::im2colGPU(float *kernel_out, float *im2col_out) {
    float *d_data;

    CHECK_CUDA( cudaMalloc((void **)&d_data, data->getTotalSize() * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(d_data, data->getTensor(), data->getTotalSize() * sizeof(float), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(kernel_out, kernel->getTensor(), kernel->getTotalSize() * sizeof(float), cudaMemcpyHostToDevice) )

    im2col_gpu<float>(d_data, data->getN(), data->getC(), data->getH(), data->getW(),
                     kernel->getH(), kernel->getW(),
                     padding, padding,
                     stride, stride,
                     dilation, dilation, im2col_out);

    CHECK_CUDA(cudaFree(d_data))
}

Tensor4d *ConvParam::col2imGPU(float *col) {
    float *d_im;
    CUDA_CHECK( cudaMalloc((void **)&d_im, getM() * getN() * sizeof(float)) )

    int num_kernels = getM();
    im2col_rev_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, col, data->getN(), kernel->getN(), getOutHeight(), getOutWidth(), d_im);

    auto im = new float[getM() * getN()];
    CUDA_CHECK( cudaMemcpy(im, d_im, getM() * getN() * sizeof(float), cudaMemcpyDeviceToHost) )

    CUDA_CHECK( cudaFree(d_im) )

    return new Tensor4d(im, data->getN(), kernel->getN(), getOutHeight(), getOutWidth());
}

bool ConvParam::checkIm2col() {
    MatrixParam *param = im2col();
    float *d_kernel, *d_im;

    cudaMalloc((void **)&d_kernel, sizeof(float) * kernel->getTotalSize());
    cudaMalloc((void**)&d_im, sizeof(float) * getIm2colSize());

    im2colGPU(d_kernel, d_im);

    auto h_A = new float[getIm2colSize()];
    auto h_B = new float[kernel->getTotalSize()];
    cudaMemcpy(h_A, d_im, sizeof(float) * getIm2colSize(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_kernel, sizeof(float) * kernel->getTotalSize(), cudaMemcpyDeviceToHost);

    h_A = transpose<float>(h_A, getK(), getM());
    h_B = transpose<float>(h_B, getN(), getK());

    cudaFree(d_kernel);
    cudaFree(d_im);

    bool ret = cmpMatrix(h_A, param->getMatA(), getM() * getK()) &&
               cmpMatrix(h_B, param->getMatB(), getN() * getK());

    delete param;
    delete[] h_A;
    delete[] h_B;
    return ret;
}











