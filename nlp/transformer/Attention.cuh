//
// Created by dbettkk on 2022/7/23.
//

#ifndef SPARSECONV_ATTENTION_CUH
#define SPARSECONV_ATTENTION_CUH

#include "interface/kernels_transformer.cuh"
#include "MatrixHalf.cuh"

class Attention {
    int heads = 8;
    int embedding = 512;
public:
    Attention();

    void forward(MatrixHalf *inputQ, MatrixHalf *inputK, MatrixHalf *inputV, MatrixHalf *output, int layer, int which_part, int *mask);

    void attn(half *Q, half *K, half *V, half *out, int batch, int en_max_len, int de_max_len, int ebd, int *mask);

    void attn_batch(half *Q, half *K, half *V, half *out, int batch, int en_max_len, int de_max_len, int* masked);

    void make_mask1(int max_len, int *out);

    void make_mask2(int max_len, int *out);
};


#endif //SPARSECONV_ATTENTION_CUH
