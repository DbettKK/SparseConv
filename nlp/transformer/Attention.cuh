//
// Created by dbettkk on 2022/7/23.
//

#ifndef SPARSECONV_ATTENTION_CUH
#define SPARSECONV_ATTENTION_CUH

#include "kernels_transformer.cuh"
#include "MatrixHalf.cuh"

class Attention {
    int heads = 8;
    int embedding = 512;
    int *mask1{}, *mask2{};
public:
    Attention(int max_len);

    void forward(MatrixHalf *inputQ, MatrixHalf *inputK, MatrixHalf *inputV, MatrixHalf *output, int layer, int which_part);

    void attn(half *Q, half *K, half *V, half *out, int batch, int max_len, int ebd, bool isMasked);

    void make_mask1(int max_len, int *out);

    void make_mask2(int max_len, int *out);
};


#endif //SPARSECONV_ATTENTION_CUH
