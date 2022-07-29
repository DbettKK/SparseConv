//
// Created by dbettkk on 2022/7/23.
//

#include "Attention.cuh"

/* 默认输入输出都为device端 */
void Attention::forward(MatrixHalf *input, MatrixHalf *output) {
    int sen = input->getRow(), ebd = input->getCol();
    // 1. input通过矩阵乘得到Q K V     input:[batch, sen, ebd]  QKV: [batch, sen, ebd]
    auto Q = new MatrixHalf(1, sen, ebd, true);
    auto K = new MatrixHalf(1, sen, ebd, true);
    auto V = new MatrixHalf(1, sen, ebd, true);
    input->gemm(this->Wq, Q);
    input->gemm(this->Wk, K);
    input->gemm(this->Wv, V);
    // 2. Q K V多头机制下分别进行运算
    // 多头机制计算本质是reshape    outQ: [batch, heads, sen, ebd / heads]
    auto outQ = new MatrixHalf(1, sen * heads, ebd / heads, true);
    auto outK = new MatrixHalf(1, sen * heads, ebd / heads, true);
    auto outV = new MatrixHalf(1, sen * heads, ebd / heads, true);
    Q->reshape(outQ, this->heads);
    K->reshape(outK, this->heads);
    V->reshape(outV, this->heads);
    Q->free_matrix();
    K->free_matrix();
    V->free_matrix();
    // softmax(QK^T)V
    auto concat = new MatrixHalf(1, sen, ebd, true);
    for (int i = 0; i < heads; i++) {
        // 1. K转置
        auto transK = new MatrixHalf(1, ebd / heads, sen, true);
        int each_block = sen * ebd / heads;
        dim3 thread(sen, ebd / heads);
        transpose_half<<<1, thread>>>(outK->getMatrix() + i * each_block, transK->getMatrix(), sen, ebd / heads);
        // 2. 计算QK^T 并softmax
        auto tmp_ans = new MatrixHalf(1, sen, sen, true);
        cublas_gemm_device(outQ->getMatrix() + i * each_block, transK->getMatrix(),
                           sen, ebd / heads, sen, tmp_ans->getMatrix());
        tmp_ans->print("tmp_ans:", true);
        tmp_ans->softmax();
        // 3. 和V再次矩阵乘
        cublas_gemm_device(tmp_ans->getMatrix(), outV->getMatrix() + i * each_block,
                           sen, sen, ebd / heads, concat->getMatrix() + i * each_block);
        // 4. cuda free
        transK->free_matrix();
        tmp_ans->free_matrix();
    }
    // 3. 运算结果concat并和 W0 运算得到输出
    auto attention_out = new MatrixHalf(1, sen, ebd, true);
    concat->gemm(this->W0, attention_out);
    // 4. 输出通过mlp层 两次矩阵乘得到最终输出
    auto out_ff = new MatrixHalf(1, sen, d_ff, true);
    //auto out_model = new MatrixHalf(1, sen, d_model, true);
    attention_out->gemm(this->Wff, out_ff);
    out_ff->gemm(this->Wm, output);

    concat->free_matrix();
    attention_out->free_matrix();
    out_ff->free_matrix();
}

void Attention::initW() {
    Wq = new MatrixHalf(1, embedding, embedding, true, 0.05);
    Wk = new MatrixHalf(1, embedding, embedding, true, 0.05);
    Wv = new MatrixHalf(1, embedding, embedding, true, 0.05);
    W0 = new MatrixHalf(1, embedding, embedding, true, 0.05);
    Wff = new MatrixHalf(1, embedding, d_ff, true, 0.05);
    Wm = new MatrixHalf(1, d_ff, d_model, true, 0.05);
}

