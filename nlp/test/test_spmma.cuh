//
// Created by dbettkk on 2022/8/14.
//

#ifndef SPARSECONV_TEST_SPMMA_CUH
#define SPARSECONV_TEST_SPMMA_CUH

#include <random>
#include "Test.cuh"
#include "../transformer/interface/spmma_interface.cuh"
#include "../transformer/interface/cublas_interface.cuh"
#include "../transformer/interface/cusparse_interface.cuh"

void test_pad_time();

void test_cusparse();

#endif //SPARSECONV_TEST_SPMMA_CUH
