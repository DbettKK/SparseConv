//
// Created by dbettkk on 2022/8/1.
//

#include "Decoder.cuh"

void Decoder::init() {
    self_attn = new Attention();
    src_attn = new Attention();
    mlp = new FeedForward();
}

void Decoder::forward(MatrixHalf *input, MatrixHalf *encoder_in, MatrixHalf *output, int layer) {
    auto self_attn_out = new MatrixHalf(input->getBatch(), input->getRow(), input->getCol(), true);
    self_attn->forward(input, input, input, self_attn_out, layer, 2);
    auto src_attn_out = new MatrixHalf(self_attn_out->getBatch(), self_attn_out->getRow(), self_attn_out->getCol(), true);
    src_attn->forward(self_attn_out, encoder_in, encoder_in, src_attn_out, layer, 3);
    mlp->forward(src_attn_out, output, layer, false);
    src_attn_out->free_matrix();
    self_attn_out->free_matrix();
}

void Decoder::forwardN(MatrixHalf *input, MatrixHalf *encoder_in, MatrixHalf *output, int N) {
    for (int i = 0; i < N; i++) {
        auto out = new MatrixHalf(input->getBatch(), input->getRow(), input->getCol(), true);
        forward(input, encoder_in, out, i);
        out->copyTo(input);
        out->free_matrix();
    }
    input->copyTo(output);
    //input->free_matrix();
}
