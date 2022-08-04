//
// Created by dbettkk on 2022/8/4.
//

#ifndef SPARSECONV_TESTS_CUH
#define SPARSECONV_TESTS_CUH

#include<cuda_fp16.h>

#include"../interface/interfaces.cuh"
#include "../net/entity/MyTensor.cuh"
#include "../net/MyResNet.cuh"

void test_bn();

void test_pool();

void test_conv2d();

void test_im2col();

#endif //SPARSECONV_TESTS_CUH
