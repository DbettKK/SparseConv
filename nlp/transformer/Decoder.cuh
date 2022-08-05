//
// Created by dbettkk on 2022/8/1.
//

#ifndef SPARSECONV_DECODER_CUH
#define SPARSECONV_DECODER_CUH


#include "Attention.cuh"
#include "FeedForward.cuh"

class Decoder {
    Attention *self_attn, *src_attn;
    FeedForward *mlp;
public:
    void init();
    void forward(MatrixHalf *input, MatrixHalf *encoder_in, MatrixHalf *output, int layer);
    void forwardN(MatrixHalf *input, MatrixHalf *encoder_in, MatrixHalf *output, int N);
};


#endif //SPARSECONV_DECODER_CUH
