//
// Created by dbettkk on 2022/8/1.
//

#ifndef SPARSECONV_TRANSFORMER_CUH
#define SPARSECONV_TRANSFORMER_CUH


#include "MatrixHalf.cuh"
#include "Encoder.cuh"
#include "Decoder.cuh"
#include <cmath>

class Transformer {
    MatrixHalf *pe, *mask1;
    MatrixHalf *W_ebd;
    Encoder *encoder;
    Decoder *decoder;

public:
    void PositionalEncoding(MatrixHalf *in, MatrixHalf *out);

    void Embedding(const int *in, int batch, int max_len, MatrixHalf *out);

    static void make_pe(int max_len, int d_model, MatrixHalf *out);

    static void make_mask1(int max_len, MatrixHalf *out);

    void init(int max_len, int d_model);

    void forward(int *en_in, int en_batch, int en_max_len, int *de_in, int de_batch, int de_max_len, int d_model,
                 MatrixHalf *out);
};


#endif //SPARSECONV_TRANSFORMER_CUH
