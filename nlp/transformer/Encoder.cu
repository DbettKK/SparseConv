//
// Created by dbettkk on 2022/7/23.
//

#include "Encoder.cuh"

void Encoder::forward(MatrixHalf *input, MatrixHalf *output) {
    auto attn_out = new MatrixHalf(input->getBatch(), input->getRow(), input->getCol(), true);
    attn->forward(input, input, input, attn_out);
    mlp->forward(attn_out, output);
    attn_out->free_matrix();
}

void Encoder::init() {
    attn = new Attention();
    mlp = new FeedForward();
    attn->initW();
    mlp->init();
}

void Encoder::free() {
    attn->free();
    mlp->free();
}