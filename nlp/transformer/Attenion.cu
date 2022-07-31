//
// Created by dbettkk on 2022/7/23.
//

#include "Attention.cuh"

/* 默认输入输出都为device端 */
void Attention::forward(MatrixHalf *inputQ, MatrixHalf *inputK, MatrixHalf *inputV, MatrixHalf *output) {
    int sen = inputQ->getRow(), ebd = inputQ->getCol();
    // 1. input通过矩阵乘得到Q K V     input:[batch, sen, ebd]  QKV: [batch, sen, ebd]
    auto Q = new MatrixHalf(1, sen, ebd, true);
    auto K = new MatrixHalf(1, sen, ebd, true);
    auto V = new MatrixHalf(1, sen, ebd, true);
    inputQ->gemm(this->Wq, Q);
    inputK->gemm(this->Wk, K);
    inputV->gemm(this->Wv, V);
    // 2. QKV [batch, sen, ebd] -> [batch, head, sen, ebd / head]
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
    // 3. 多头注意力
    auto concat = new MatrixHalf(1, sen, ebd, true);
    attn(outQ->getMatrix(), outK->getMatrix(), outV->getMatrix(), concat->getMatrix(), sen, ebd);

    // 4. 再一个线性层 运算结果concat并和 W0 运算得到输出
    auto attention_out = new MatrixHalf(1, sen, ebd, true);
    concat->gemm(this->W0, attention_out);
    concat->print("concat:", true);

    // 5. add+layerNorm
    concat->addMatrix(inputQ, output);

    // 6. free
    concat->free_matrix();
    attention_out->free_matrix();

}

void Attention::initW() {
    Wq = new MatrixHalf(1, embedding, embedding, true, 0.05);
    Wk = new MatrixHalf(1, embedding, embedding, true, 0.05);
    Wv = new MatrixHalf(1, embedding, embedding, true, 0.05);
    W0 = new MatrixHalf(1, embedding, embedding, true, 0.05);
}

void Attention::attn(half *Q, half *K, half *V, half *out, int max_len, int ebd) const {
    half *transK, *ans;
    int *mask;
    cudaMalloc(&transK, sizeof(half) * max_len * ebd / heads);
    cudaMalloc(&ans, sizeof(half) * max_len * max_len);
    cudaMalloc(&mask, sizeof(int) * max_len * max_len);
    cudaMemset(mask, 0, sizeof(int) * max_len * max_len);
    dim3 block(max_len, ebd / heads);
    for (int i = 0; i < heads; i++) {
        int each_block = max_len * ebd / heads;
        // 1. K转置
        transpose_half<<<1, block>>>(K + i * each_block, transK, max_len, ebd / heads);
        // 2. QK^T
        cublas_gemm_device(Q + i * each_block, transK, max_len, ebd / heads, max_len, ans);
        // 3. mask
        mask_matrix_gpu<<<max_len, max_len>>>(ans, mask, max_len, max_len);
        // 4. softmax
        softmax_half<<<max_len, max_len>>>(ans, max_len, max_len);
        // MatrixHalf::print_device(ans, max_len, max_len);
        // 5. 和V乘
        cublas_gemm_device(ans, V + i * each_block, max_len, max_len, ebd / heads, out + i * each_block);
    }
    // free
    cudaFree(transK);
    cudaFree(ans);
}

void Attention::free() {
    W0->free_matrix();
    Wq->free_matrix();
    Wk->free_matrix();
    Wv->free_matrix();
}

