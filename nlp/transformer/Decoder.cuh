//
// Created by dbettkk on 2022/8/1.
//

#ifndef SPARSECONV_DECODER_CUH
#define SPARSECONV_DECODER_CUH


#include "Attention.cuh"
#include "FeedForward.cuh"

class Decoder {
    int *mask1, *mask2;
    Attention *self_attn, *src_attn;
    FeedForward *mlp;
public:
    Decoder(int *mask1, int *mask2);
    void forward(MatrixHalf *input, MatrixHalf *encoder_in, MatrixHalf *output, int layer);
    void forwardN(MatrixHalf *input, MatrixHalf *encoder_in, MatrixHalf *output, int N);
};


#endif //SPARSECONV_DECODER_CUH
