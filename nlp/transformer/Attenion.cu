//
// Created by dbettkk on 2022/7/23.
//

#include "Attention.cuh"

/* 默认输入输出都为device端 */
void Attention::forward(MatrixHalf *input) {
    int sen = input->getRow(), ebd = input->getCol();
    // 1. input通过矩阵乘得到Q K V     input:[batch, sen, ebd]  QKV: [batch, sen, ebd]
    auto Q = new MatrixHalf(1, input->getRow(), input->getCol(), true);
    auto K = new MatrixHalf(1, input->getRow(), input->getCol(), true);
    auto V = new MatrixHalf(1, input->getRow(), input->getCol(), true);
    input->gemm(this->Wq, Q);
    input->gemm(this->Wk, K);
    input->gemm(this->Wv, V);
    // 2. Q K V多头机制下分别进行运算
    // 多头机制计算本质是reshape    outQ: [batch, heads, sen, ebd / heads]
    auto outQ = new MatrixHalf(1, Q->getRow(), Q->getCol(), true);
    auto outK = new MatrixHalf(1, K->getRow(), K->getCol(), true);
    auto outV = new MatrixHalf(1, V->getRow(), V->getCol(), true);
    Q->reshape(outQ, this->heads);
    K->reshape(outK, this->heads);
    V->reshape(outV, this->heads);
    // softmax(QK^T)V
    auto concat = new MatrixHalf(1, sen, ebd, true);
    for (int i = 0; i < heads; i++) {
        // 1. K转置
        auto transK = new MatrixHalf(1, K->getCol() / heads, K->getRow(), true);
        int each_block = transK->getCol() * transK->getRow();
        dim3 thread(transK->getCol(), transK->getRow());
        transpose_half<<<1, thread>>>(outK->getMatrix() + i * each_block,
                                      transK->getMatrix(), transK->getCol(), transK->getRow());
        // 2. 计算QK^T 并softmax
        auto tmp_ans = new MatrixHalf(1, transK->getCol(),  transK->getCol(), true);
        cublas_gemm_device(outQ->getMatrix() + i * each_block, transK->getMatrix(),
                           transK->getCol(), transK->getRow(), transK->getCol(), tmp_ans->getMatrix());
        tmp_ans->softmax();
        // 3. 和V再次矩阵乘
        cublas_gemm_device(tmp_ans->getMatrix(), outV->getMatrix() + i * each_block,
                           tmp_ans->getRow(), tmp_ans->getCol(), transK->getRow(), concat->getMatrix() + i * each_block);
    }
    // 3. 运算结果concat并和 W0 运算得到输出
    auto attention_out = new MatrixHalf(1, sen, ebd, true);
    concat->gemm(this->W0, attention_out);
    // 4. 输出通过mlp层 两次矩阵乘得到最终输出

}

void Attention::initW() {
    Wq = new MatrixHalf(1, embedding, embedding, true);
    Wk = new MatrixHalf(1, embedding, embedding, true);
    Wv = new MatrixHalf(1, embedding, embedding, true);
    W0 = new MatrixHalf(1, embedding, embedding, true);
    Wff = new MatrixHalf(1, embedding, d_ff, true);
    Wm = new MatrixHalf(1, d_ff, d_model, true);
}

