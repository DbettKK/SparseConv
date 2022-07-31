//
// Created by dbettkk on 2022/7/31.
//

#include "FeedForward.cuh"

void FeedForward::init() {
    W1 = new MatrixHalf(1, d_model, d_ff, true, 0.03);
    W2 = new MatrixHalf(1, d_ff, d_model, true, 0.02);
}

void FeedForward::forward(MatrixHalf *input, MatrixHalf *output) {
    // 1. 声明所需变量
    auto ff = new MatrixHalf(input->getBatch(), input->getRow(), d_ff, true);
    auto mo = new MatrixHalf(input->getBatch(), input->getRow(), d_model, true);
    // 第一个线性层
    input->gemm(this->W1, ff);
    // relu
    ff->relu();
    // 第二个线性层
    ff->gemm(W2, mo);
    // Add & LayerNorm
    ff->addMatrix(input, output);
}

void FeedForward::free() {
    W1->free_matrix();
    W2->free_matrix();
}
