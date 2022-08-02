//
// Created by dbettkk on 2022/7/23.
//

#include "Attention.cuh"

/* 默认输入输出都为device端 */
void Attention::forward(MatrixHalf *inputQ, MatrixHalf *inputK, MatrixHalf *inputV, MatrixHalf *output) {
    // 1. input通过矩阵乘得到Q K V     input:[batch, sen, ebd]  QKV: [batch, sen, ebd]
    auto Q = new MatrixHalf(inputQ->getBatch(), inputQ->getRow(), inputQ->getCol(), true);
    auto K = new MatrixHalf(inputK->getBatch(), inputK->getRow(), inputK->getCol(), true);
    auto V = new MatrixHalf(inputV->getBatch(), inputV->getRow(), inputV->getCol(), true);
    inputQ->gemm_batches(this->Wq, Q, true);
    inputK->gemm_batches(this->Wk, K, true);
    inputV->gemm_batches(this->Wv, V, true);
    // 2. QKV [batch, sen, ebd] -> [batch, head, sen, ebd / head]
    // 多头机制计算本质是reshape    outQ: [batch, heads, sen, ebd / heads]
    auto outQ = new MatrixHalf(Q->getBatch(), Q->getRow(), Q->getCol(), true);
    auto outK = new MatrixHalf(K->getBatch(), K->getRow(), K->getCol(), true);
    auto outV = new MatrixHalf(V->getBatch(), V->getRow(), V->getCol(), true);
    Q->reshape(outQ, this->heads);
    K->reshape(outK, this->heads);
    V->reshape(outV, this->heads);
    Q->free_matrix();
    K->free_matrix();
    V->free_matrix();
    // 3. 多头注意力
    auto concat = new MatrixHalf(inputQ->getBatch(), inputQ->getRow(), inputQ->getCol(), true);
    attn(outQ->getMatrix(), outK->getMatrix(), outV->getMatrix(), concat->getMatrix(),
         inputQ->getBatch(), inputQ->getRow(), inputQ->getCol());

    // 4. 再一个线性层 运算结果concat并和 W0 运算得到输出
    auto attention_out = new MatrixHalf(inputQ->getBatch(), inputQ->getRow(), inputQ->getCol(), true);
    concat->gemm_batches(this->W0, attention_out, true);
    //concat->print("concat:", true);

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

void Attention::attn(half *Q, half *K, half *V, half *out, int batch, int max_len, int ebd) {
    half *transK, *ans;
    int *mask;
    cudaMalloc(&mask, sizeof(int) * max_len * max_len);
    this->make_mask1(max_len, mask);
    cudaMalloc(&transK, sizeof(half) * max_len * ebd / heads);
    cudaMalloc(&ans, sizeof(half) * max_len * max_len);
    dim3 block(max_len, ebd / heads);
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < heads; i++) {
            int each_block = (b * heads + i) * max_len * ebd / heads;
            // 1. K转置
            transpose_half<<<1, block>>>(K + each_block, transK, max_len, ebd / heads);
            // 2. QK^T
            cublas_gemm_device(Q + each_block, transK, max_len, ebd / heads, max_len, ans);
            // 3. mask
            mask_matrix_gpu<<<max_len, max_len>>>(ans, mask, max_len, max_len);
            // 4. softmax
            softmax_half<<<max_len, max_len>>>(ans, max_len, max_len);
            // MatrixHalf::print_device(ans, max_len, max_len);
            // 5. 和V乘
            cublas_gemm_device(ans, V + each_block, max_len, max_len, ebd / heads, out + each_block);
        }
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

void Attention::make_mask1(int max_len, int *out) {
    // 从主对角线开始 隔两个对角线的值不mask
    int *h_mask = new int[max_len * max_len];
    memset(h_mask, 0, sizeof(int) * max_len * max_len);
    int max_num = (max_len - 1) / 3;
    for (int i = 0; i < max_len; i++) {
        for (int j = 0; j < max_len; j++) {
            for (int k = 0; k <= max_num; k++) {
                if (i == j + k * 3) h_mask[i * max_len + j] = 1;
                if (j == i + k * 3) h_mask[i * max_len + j] = 1;
            }
        }
    }
    cudaMemcpy(out, h_mask, sizeof(int) * max_len * max_len, cudaMemcpyHostToDevice);
}

