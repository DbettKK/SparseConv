//
// Created by dbettkk on 2022/8/1.
//

#ifndef SPARSECONV_TRANSFORMER_CUH
#define SPARSECONV_TRANSFORMER_CUH


#include "MatrixHalf.cuh"
#include <cmath>

class Transformer {
    MatrixHalf *pe;

public:
    void PositionalEncoding(MatrixHalf *in, MatrixHalf *out);

    static void make_pe(int max_len, int d_model, MatrixHalf *out);

    static void make_mask1(int max_len, MatrixHalf *out);

    void init(int max_len, int d_model);
};


#endif //SPARSECONV_TRANSFORMER_CUH
