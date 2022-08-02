//
// Created by dbettkk on 2022/8/1.
//

#include "Decoder.cuh"

void Decoder::init() {
    self_attn = new Attention();
    src_attn = new Attention();
    mlp = new FeedForward();
    self_attn->initW();
    src_attn->initW();
    mlp->init();
}

void Decoder::free() {
    src_attn->free();
    self_attn->free();
    mlp->free();
}

void Decoder::forward(MatrixHalf *input, MatrixHalf *encoder_in, MatrixHalf *output) {
    auto self_attn_out = new MatrixHalf(input->getBatch(), input->getRow(), input->getCol(), true);
    self_attn->forward(input, input, input, self_attn_out);
    auto src_attn_out = new MatrixHalf(self_attn_out->getBatch(), self_attn_out->getRow(), self_attn_out->getCol(), true);
    src_attn->forward(self_attn_out, encoder_in, encoder_in, src_attn_out);
    mlp->forward(src_attn_out, output);
    src_attn_out->free_matrix();
    self_attn_out->free_matrix();
}

void Decoder::forwardN(MatrixHalf *input, MatrixHalf *encoder_in, MatrixHalf *output, int N) {
    for (int i = 0; i < N; i++) {
        auto out = new MatrixHalf(input->getBatch(), input->getRow(), input->getCol(), true);
        forward(input, encoder_in, out);
        out->copyTo(input);
        out->free_matrix();
    }
    input->copyTo(output);
    //input->free_matrix();
}
