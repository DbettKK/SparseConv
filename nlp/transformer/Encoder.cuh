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
    void free();
    void forward(MatrixHalf *input, MatrixHalf *output);
};


#endif //SPARSECONV_ENCODER_CUH
