//
// Created by dbettkk on 2022/7/31.
//

#ifndef SPARSECONV_FEEDFORWARD_CUH
#define SPARSECONV_FEEDFORWARD_CUH


#include "MatrixHalf.cuh"

class FeedForward {
    MatrixHalf *W1, *W2;
    int d_ff = 2048, d_model = 512;
public:
    void init();
    void free();
    void forward(MatrixHalf *input, MatrixHalf *output);
};


#endif //SPARSECONV_FEEDFORWARD_CUH
