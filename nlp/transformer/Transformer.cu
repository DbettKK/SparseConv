//
// Created by dbettkk on 2022/8/1.
//

#include "Transformer.cuh"

void Transformer::PositionalEncoding(MatrixHalf *in, MatrixHalf *out) {
    in->addMatrix(pe, out);
}

void Transformer::make_pe(int max_len, int d_model, MatrixHalf *out) {
    auto div_term = new double[d_model / 2 + 1];
    for (int i = 0; i < d_model / 2 + 1; i++) {
        div_term[i] = exp(i * 2 * -log(10000.0) / d_model);
    }
    half *pe = new half[max_len * d_model];
    for (int i = 0; i < max_len; i++) {
        for (int j = 0; j < d_model; j++) {
            int idx = i * d_model + j;
            if (j % 2 == 0) {
                pe[idx] = sin(div_term[j / 2]);
            } else {
                pe[idx] = cos(div_term[j / 2]);
            }
        }
    }
    half *d_pe;
    cudaMalloc(&d_pe, sizeof(half) * max_len * d_model);
    cudaMemcpy(d_pe, pe, sizeof(half) * max_len * d_model, cudaMemcpyHostToDevice);
    out->setMatrix(d_pe);

    delete[] div_term;
    delete[] pe;
}

void Transformer::init(int max_len, int d_model) {
    pe = new MatrixHalf(1, max_len, d_model, true);
    mask1 = new MatrixHalf(1, max_len, max_len, true);
    make_pe(max_len, d_model, pe);
    //make_mask1(max_len, mask1);
    W_ebd = new MatrixHalf(1, max_len, d_model, true, 1);
    encoder = new Encoder();
    encoder->init();
    decoder = new Decoder();
    decoder->init();
}

void Transformer::make_mask1(int max_len, MatrixHalf *out) {
    // 从主对角线开始 隔两个对角线的值不mask
    half *h_mask = new half[max_len * max_len];
    memset(h_mask, 0, sizeof(half) * max_len * max_len);
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
            printf("%d ", __half2int_rz(h_mask[i * max_len + j]));
        }
        printf("\n");
    }
}

void Transformer::Embedding(const int *in, int batch, int max_len, MatrixHalf *out) {
    // in: [batch, max_len] type: INT/LONG  HOST
    // out: [batch, max_len, embedding_dim]
    int ebd_dim = out->getCol();
    half *one_hot = new half[batch * max_len * max_len];
    memset(one_hot, 0, sizeof(half) * batch * max_len * max_len);
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < max_len; j++) {
            int idx = i * max_len * max_len + j * max_len + in[i * max_len + j];
            one_hot[idx - 1] = 1;
        }
    }
    auto dh = new MatrixHalf(batch, max_len, max_len, true);
    cudaMemcpy(dh->getMatrix(), one_hot, sizeof(half) * batch * max_len * max_len, cudaMemcpyHostToDevice);

    dh->gemm_batches(W_ebd, out, true);

    delete[] one_hot;
    dh->free_matrix();
    //delete[] one_hot;
}

void Transformer::forward(int *en_in, int en_batch, int en_max_len, int *de_in, int de_batch, int de_max_len, int d_model,
                          MatrixHalf *out) {
    // 1. embedding
    auto en_ebd_out = new MatrixHalf(en_batch, en_max_len, d_model, true);
    auto de_ebd_out = new MatrixHalf(de_batch, de_max_len, d_model, true);
    Embedding(en_in, en_batch, en_max_len, en_ebd_out);
    Embedding(de_in, de_batch, de_max_len, de_ebd_out);
    // 2. position encoding
    auto en_pe_out = new MatrixHalf(en_batch, en_max_len, d_model, true);
    auto de_pe_out = new MatrixHalf(de_batch, de_max_len, d_model, true);
    PositionalEncoding(en_ebd_out, en_pe_out);
    PositionalEncoding(de_ebd_out, de_pe_out);
    // 3. encoder
    auto encoder_out = new MatrixHalf(en_batch, en_max_len, d_model, true);
    encoder->forwardN(en_pe_out, encoder_out, 6);
    // 4. decoder
    auto decoder_out = new MatrixHalf(de_batch, de_max_len, d_model, true);
    decoder->forwardN(de_pe_out, encoder_out, decoder_out, 6);
    // 5. liner + softmax

    decoder_out->print("out:", true);

    decoder_out->copyTo(out);

    // 6. free
    en_ebd_out->free_matrix();
    de_ebd_out->free_matrix();
    en_pe_out->free_matrix();
    de_pe_out->free_matrix();
    encoder_out->free_matrix();
    decoder_out->free_matrix();
}
