//
// Created by dbettkk on 2022/7/22.
//

#ifndef SPARSECONV_NET_CUH
#define SPARSECONV_NET_CUH

#include "../interface/sparse_mma_gemm.cuh"

struct Matrix {
    half *item;
    int row, col;

    Matrix(half *item, int row, int col) : item(item), row(row), col(col) {}
};

void embedding(Matrix *in, Matrix *out, Matrix *W_ebd);

void position_encoding(Matrix *in, Matrix *out);

void encoder();

void decoder();

void attention();

void feed_forward();

#endif //SPARSECONV_NET_CUH
