//
// Created by dbettkk on 2022/7/31.
//

#ifndef SPARSECONV_FEEDFORWARD_CUH
#define SPARSECONV_FEEDFORWARD_CUH


#include "MatrixHalf.cuh"

class FeedForward {
    int d_ff = 2048, d_model = 512;
public:
    void forward(MatrixHalf *input, MatrixHalf *output, int layer, bool is_encoder);
};


#endif //SPARSECONV_FEEDFORWARD_CUH
