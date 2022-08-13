//
// Created by dbettkk on 2022/8/8.
//

#ifndef SPARSECONV_TEST_NLP_CUH
#define SPARSECONV_TEST_NLP_CUH

#include "../transformer/kernels_transformer.cuh"
#include <random>

void test_gemm_batches();

void test_spmma_cublas();

void test_spmma_cublas_efficient();

void test_spmma_batches();

void test_transpose_batches();

void test_softmax();

void test_shared_mem();
#endif //SPARSECONV_TEST_NLP_CUH
