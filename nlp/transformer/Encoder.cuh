//
// Created by dbettkk on 2022/7/23.
//

#ifndef SPARSECONV_ENCODER_CUH
#define SPARSECONV_ENCODER_CUH


#include "MatrixHalf.cuh"
#include "Attention.cuh"
#include "FeedForward.cuh"

class Encoder {
    Attention *attn;
    FeedForward *mlp;
public:
    void init();
    void forward(MatrixHalf *input, MatrixHalf *output, int layer);
    void forwardN(MatrixHalf *input, MatrixHalf *output, int N);
};


#endif //SPARSECONV_ENCODER_CUH
