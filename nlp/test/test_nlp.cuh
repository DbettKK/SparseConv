//
// Created by dbettkk on 2022/8/8.
//

#ifndef SPARSECONV_TEST_NLP_CUH
#define SPARSECONV_TEST_NLP_CUH

#include "../transformer/kernels_transformer.cuh"
#include <random>

void test_gemm_batches();

void test_spmma_cublas();

#endif //SPARSECONV_TEST_NLP_CUH
