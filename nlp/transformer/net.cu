//
// Created by dbettkk on 2022/7/22.
//

#include "net.cuh"

void embedding(Matrix *in, Matrix *out, Matrix *W_ebd) {
    cublas_gemm(in->item, W_ebd->item, in->row, in->col, W_ebd->col, out->item);
}

void position_encoding(Matrix *in, Matrix *out) {

}

void encoder(Matrix *in, Matrix *Wq, Matrix *Wk, Matrix *Wv, Matrix *out) {

}
