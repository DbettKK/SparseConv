//
// Created by dbettkk on 2022/8/14.
//

#ifndef SPARSECONV_TEST_CUH
#define SPARSECONV_TEST_CUH

#include <random>
#include "../transformer/interface/kernels_transformer.cuh"

class Test {
    int batch, m, k, n;

public:
    Test(int batch, int m, int k, int n);

    int matA_size() const;

    int matB_size() const;

    int matC_size() const;

    float randf();
    float randf(int bound);

    void generate_sparse_A(half **matA);

    void generate_dense_B(half **matB);

    void generate_zero_C(half **matC) const;

    void matC_diff(half *matC1, half *matC2) const;

    void print_matC(half *matC) const;
};


#endif //SPARSECONV_TEST_CUH
