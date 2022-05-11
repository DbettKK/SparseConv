//
// Created by dbettkk on 2022/3/31.
//

#ifndef SPARSECONVOLUTION_SPARSE_CONV_CUH
#define SPARSECONVOLUTION_SPARSE_CONV_CUH

#include "sparse_matmul.cuh"

/**
 * param: host
 * return: host
 */
Tensor4d *sparse_conv(ConvParam *param);

#endif //SPARSECONVOLUTION_SPARSE_CONV_CUH
