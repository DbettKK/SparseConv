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
    MatrixHalf *Wff, *Wm;
    int heads = 8;
    int embedding = 512;
    int d_ff = 2048, d_model = 512;
public:
    void forward(MatrixHalf *input, MatrixHalf *output);

    void initW();
};


#endif //SPARSECONV_ATTENTION_CUH
