//
// Created by dbettkk on 2022/7/31.
//

#include "FeedForward.cuh"

void FeedForward::forward(MatrixHalf *input, MatrixHalf *output, int layer, bool is_encoder) {
    std::string path_ff = is_encoder ? "../../data/transformer/wff_en" + std::to_string(layer) :
                          "../../data/transformer/wff_de" + std::to_string(layer);
    std::string path_m = is_encoder ? "../../data/transformer/wm_en" + std::to_string(layer) :
                          "../../data/transformer/wm_de" + std::to_string(layer);
    // 1. 声明所需变量
    auto ff = new MatrixHalf(input->getBatch(), input->getRow(), d_ff, true);
    auto mo = new MatrixHalf(input->getBatch(), input->getRow(), d_model, true);
    // 第一个线性层
    auto Wff = new MatrixHalf(1, d_model, d_ff, true, path_ff);
    input->gemm_batches(Wff, ff, true);
    // relu
    ff->relu();
    // 第二个线性层
    auto Wm = new MatrixHalf(1, d_ff, d_model, true, path_m);
    ff->gemm_batches(Wm, mo, true);
    // Add & LayerNorm
    auto add_out = new MatrixHalf(input->getBatch(), input->getRow(), input->getCol(), true);
    mo->addMatrix(input, add_out);
    add_out->layerNorm(output);
    // free
    Wff->free_matrix();
    Wm->free_matrix();
    add_out->free_matrix();
}

