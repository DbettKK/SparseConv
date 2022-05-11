//
// Created by dbettkk on 2022/3/31.
//

#ifndef CUDNNCONV_CONST_CUH
#define CUDNNCONV_CONST_CUH

const int M = 16;
const int N = 16;
const int K = 16;

const int DATA_N = 4, DATA_C = 3, DATA_H = 16, DATA_W = 16;
const int KERNEL_N = 4, KERNEL_C = 3, KERNEL_H = 3, KERNEL_W = 3;

const int PADDING = 0;
const int STRIDE = 1;
const int DILATION = 1;

const std::string CUDNN_DATA_PATH = "../../data/data.bin";  /* NOLINT */
const std::string CUDNN_KERNEL_PATH = "../../data/kernel.bin"; /* NOLINT */


#endif //CUDNNCONV_CONST_CUH
