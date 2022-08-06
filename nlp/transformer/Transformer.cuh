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
    MatrixHalf *pe;
    MatrixHalf *W_ebd, *W_last;
    Encoder *encoder;
    Decoder *decoder;
    const int batch, en_max_len, de_max_len, d_model;

public:
    Transformer(int batch, int enMaxLen, int deMaxLen, int dModel);

    void PositionalEncoding(MatrixHalf *in, MatrixHalf *out);

    void Embedding(const int *in, int batch, int max_len, MatrixHalf *out);

    static void make_pe(int batch, int max_len, int d_model, MatrixHalf *out);

    void init(int batch, int max_len, int d_model);

    void forward(int *en_in, int *de_in, MatrixHalf *out);
};


#endif //SPARSECONV_TRANSFORMER_CUH
