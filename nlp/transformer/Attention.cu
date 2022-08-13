//
// Created by dbettkk on 2022/7/23.
//

#include "Attention.cuh"

std::string getLayerString(int which_part) {
    std::string en_or_de;
    switch (which_part) {
        case 1: en_or_de = "en";
            break;
        case 2: en_or_de = "self_de";
            break;
        case 3: en_or_de = "src_de";
            break;
        default: en_or_de = "en";
    }
    return en_or_de;
}

/* 默认输入输出都为device端 */
void Attention::forward(MatrixHalf *inputQ, MatrixHalf *inputK, MatrixHalf *inputV, MatrixHalf *output, int layer, int which_part) {
    std::string path_suffix = getLayerString(which_part) +  std::to_string(layer);
    // 1. input通过矩阵乘得到Q K V     input:[batch, sen, ebd]  QKV: [batch, sen, ebd]
    auto Q = new MatrixHalf(inputQ->getBatch(), inputQ->getRow(), inputQ->getCol(), true);
    auto K = new MatrixHalf(inputK->getBatch(), inputK->getRow(), inputK->getCol(), true);
    auto V = new MatrixHalf(inputV->getBatch(), inputV->getRow(), inputV->getCol(), true);
    auto Wq = new MatrixHalf(1, embedding, embedding, true, "../../data/transformer/wq_" + path_suffix);
    auto Wk = new MatrixHalf(1, embedding, embedding, true, "../../data/transformer/wk_" + path_suffix);
    auto Wv = new MatrixHalf(1, embedding, embedding, true, "../../data/transformer/wv_" + path_suffix);
    inputQ->gemm_batches(Wq, Q, true);
    inputK->gemm_batches(Wk, K, true);
    inputV->gemm_batches(Wv, V, true);
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
//    attn(outQ->getMatrix(), outK->getMatrix(), outV->getMatrix(), concat->getMatrix(),
//         inputQ->getBatch(), inputK->getRow(), inputQ->getRow(), inputQ->getCol(), which_part == 3);
    attn_batch(outQ->getMatrix(), outK->getMatrix(), outV->getMatrix(), concat->getMatrix(),
         inputQ->getBatch(), inputK->getRow(), inputQ->getRow(), nullptr);
    // 4. 再一个线性层 运算结果concat并和 W0 运算得到输出
    auto attention_out = new MatrixHalf(inputQ->getBatch(), inputQ->getRow(), inputQ->getCol(), true);
    auto W0 = new MatrixHalf(1, embedding, embedding, true, "../../data/transformer/w0_" + path_suffix);
    concat->gemm_batches(W0, attention_out, true);
    // 5. add+layerNorm
    auto add_out = new MatrixHalf(inputQ->getBatch(), inputQ->getRow(), inputQ->getCol(), true);
    attention_out->addMatrix(inputQ, add_out);
    //add_out->print("add_out", true);
    auto ln_out = new MatrixHalf(inputQ->getBatch(), inputQ->getRow(), inputQ->getCol(), true);
    add_out->layerNorm(ln_out);
    //ln_out->print("ln_out", true);

    ln_out->copyTo(output);
    // 6. free
    concat->free_matrix();
    attention_out->free_matrix();
    add_out->free_matrix();
    ln_out->free_matrix();
    Wq->free_matrix();
    Wk->free_matrix();
    Wv->free_matrix();
    W0->free_matrix();
}

void Attention::attn(half *Q, half *K, half *V, half *out, int batch, int en_max_len, int de_max_len, int ebd, bool isMasked) {
    // self-attn: en_max_len == de_max_len
    // attn: en_max_len != de_max_len
    half *transK, *ans, *softmax_out;
    cudaMalloc(&transK, sizeof(half) * en_max_len * ebd / heads);
    cudaMalloc(&ans, sizeof(half) * de_max_len * en_max_len);
    cudaMalloc(&softmax_out, sizeof(half) * de_max_len * en_max_len);
    dim3 grid(en_max_len / 32 + 1, ebd / heads / 32 + 1);
    dim3 block(32, 32);
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < heads; i++) {
            int each_block = (b * heads + i) * ebd / heads;
            // 1. K转置
            transpose_half<<<grid, block>>>(K + each_block * en_max_len, transK, en_max_len, ebd / heads);
            // 2. QK^T / sqrt(d_k)
            cublas_gemm_device_scale(Q + each_block * de_max_len, transK, de_max_len, ebd / heads, en_max_len, 1.0f / (float)sqrt(ebd), ans);
            // 3. mask
            mask_matrix_gpu<<<en_max_len * de_max_len / 32 + 1, 32>>>(ans, isMasked ? mask2 : mask1, de_max_len, en_max_len);
            //MatrixHalf::print_device(ans, de_max_len, en_max_len);
            // 4. softmax
            //softmax_cudnn_trans(ans, de_max_len, en_max_len, 1, 1, softmax_out);
            softmax_half<<<de_max_len, en_max_len>>>(ans, de_max_len, en_max_len, softmax_out);
            // 5. 和V乘
            //half *tmp;
            //CHECK_CUDA(cudaMalloc(&tmp, sizeof(half) * de_max_len * ebd / heads))
            sparse_mma_gemm_device(softmax_out, V + each_block * en_max_len, de_max_len, en_max_len,
                                   ebd / heads, true, out + each_block * de_max_len);
            //cublas_gemm_device(softmax_out, V + each_block * en_max_len, de_max_len, en_max_len, ebd / heads, out + each_block * de_max_len);
            //cusparse_gemm_csr_device(softmax_out, V + each_block * en_max_len, de_max_len, en_max_len, ebd / heads, out + each_block * de_max_len);
            //MatrixHalf::cmp(tmp, out + each_block * de_max_len, de_max_len * ebd / heads);
        }
    }
    // free
    cudaFree(transK);
    cudaFree(ans);
    cudaFree(softmax_out);
}

void Attention::attn_batch(half *Q, half *K, half *V, half *out, int batch, int en_max_len, int de_max_len, int *mask) {
    // Q: [batch, head, de_max_len, ebd / heads]
    // K,V: [batch, head, en_max_len, ebd / heads]

    // 0. 相关参数init
    int k_size = batch * embedding * en_max_len;
    half *transK, *attn_matrix, *softmax_out;
    CHECK_CUDA(cudaMalloc(&transK, sizeof(half) * k_size))
    CHECK_CUDA(cudaMalloc(&attn_matrix, sizeof(half) * batch * heads * de_max_len * en_max_len))
    CHECK_CUDA(cudaMalloc(&softmax_out, sizeof(half) * batch * heads * de_max_len * en_max_len))

    // 1. K转置
    dim3 grid(batch * heads / 32 + 1, en_max_len, embedding / heads);
    transpose_batches<<<grid, 32>>>(K, transK, batch * heads, en_max_len, embedding / heads);

    // 2. QK^T / sqrt(d_k)
    cublas_gemm_batches_scale_device(Q, transK, batch * heads, de_max_len, embedding / heads, en_max_len, 1.0f / (float)sqrt(embedding), attn_matrix);

    // 3. mask
    if (mask != nullptr) {
        dim3 grid_mask(batch * heads / 32 + 1, de_max_len, en_max_len);
        mask_matrix_batches<<<grid_mask, 32>>>(attn_matrix, mask, batch * heads, de_max_len, en_max_len);
    }

    // 4.softmax
    dim3 grid_softmax(de_max_len, batch * heads);
    softmax_batches<<<grid_softmax, en_max_len>>>(attn_matrix, batch * heads, de_max_len, en_max_len, softmax_out);

    // 5. 和V乘
    //sparse_mma_gemm_batches_device(softmax_out, V, batch * heads, de_max_len, en_max_len, embedding / heads, true, out);
    cublas_gemm_batches_device(softmax_out, V, batch * heads, de_max_len, en_max_len, embedding / heads, false, out);

    // 6. free
    CHECK_CUDA(cudaFree(transK))
    CHECK_CUDA(cudaFree(attn_matrix))
    CHECK_CUDA(cudaFree(softmax_out))
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

void Attention::make_mask2(int max_len, int *out) {
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
    for (int i = 0; i < max_len; i++) {
        for (int j = 0; j < max_len; j++) {
            if (j > i) h_mask[i * max_len + j] = 0;
        }
    }
    cudaMemcpy(out, h_mask, sizeof(int) * max_len * max_len, cudaMemcpyHostToDevice);
}

Attention::Attention(int max_len) {
    CHECK_CUDA(cudaMalloc(&mask1, sizeof(int) * max_len * max_len))
    CHECK_CUDA(cudaMalloc(&mask2, sizeof(int) * max_len * max_len))
    make_mask1(max_len, mask1);
    make_mask2(max_len, mask2);
}



