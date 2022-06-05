//
// Created by dbettkk on 2022/3/29.
//

#ifndef SPARSECONVOLUTION_CONVPARAM_CUH
#define SPARSECONVOLUTION_CONVPARAM_CUH

#include<iostream>
#include<cuda_fp16.h>
#include "Tensor4d.cuh"
#include "MatrixParam.cuh"
#include "../kernels/kernels.cuh"

class ConvParam {
    Tensor4d *data, *kernel;
    int padding, stride, dilation;

    Tensor4d* padData();

    half* data2col();

    half *kernel2col();
public:
    ConvParam();

    ConvParam(Tensor4d *data, Tensor4d *kernel, int padding, int stride, int dilation);

    Tensor4d *getData() const;

    Tensor4d *getKernel() const;

    int getPadding() const;

    int getStride() const;

    int getDilation() const;

    int getOutHeight() const;

    int getOutWidth() const;

    int getIm2colSize() const;

    int getM() const;

    int getK() const;

    int getN() const;

    MatrixParam *im2col();

    Tensor4d *col2im(MatrixParam *param);

    /**
     * kernel_out: device kernel平铺
     * im2col_out: device im2col后的结果
     */
    void im2colGPU(half *kernel_out, half *im2col_out);

    // col: device 需要col2im的输入
    Tensor4d* col2imGPU(half *col);

    bool checkIm2col();
};


#endif //SPARSECONVOLUTION_CONVPARAM_CUH
