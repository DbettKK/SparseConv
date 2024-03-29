//
// Created by dbettkk on 2022/7/23.
//

#include "Encoder.cuh"

void Encoder::forward(MatrixHalf *input, MatrixHalf *output, int layer) {
    auto attn_out = new MatrixHalf(input->getBatch(), input->getRow(), input->getCol(), true);
    attn->forward(input, input, input, attn_out, layer, 1, mask);
    //attn_out->print("attn", true);
    mlp->forward(attn_out, output, layer, true);
    attn_out->free_matrix();
}

void Encoder::forwardN(MatrixHalf *input, MatrixHalf *output, int N) {
    for (int i = 0; i < N; i++) {
        auto out = new MatrixHalf(input->getBatch(), input->getRow(), input->getCol(), true);
        forward(input, out, i);
        out->copyTo(input);
        out->free_matrix();
    }
    input->copyTo(output);
    //input->free_matrix();
}

Encoder::Encoder(int *mask) : mask(mask) {
    attn = new Attention();
    mlp = new FeedForward();
}
