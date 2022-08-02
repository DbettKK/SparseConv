//
// Created by dbettkk on 2022/7/23.
//

#ifndef SPARSECONV_ATTENTION_CUH
#define SPARSECONV_ATTENTION_CUH

#include "kernels_transformer.cuh"
#include "MatrixHalf.cuh"

class Attention {
    MatrixHalf *Wq, *Wk, *Wv;
    MatrixHalf *W0;
    int heads = 8;
    int embedding = 512;
    int d_ff = 2048, d_model = 512;
public:
    void forward(MatrixHalf *inputQ, MatrixHalf *inputK, MatrixHalf *inputV, MatrixHalf *output);

    void initW();

    void free();

    void attn(half *Q, half *K, half *V, half *out, int batch, int max_len, int ebd);

    void make_mask1(int max_len, int *out);
};


#endif //SPARSECONV_ATTENTION_CUH
