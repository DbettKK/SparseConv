//
// Created by dbettkk on 2022/8/1.
//

#include "Transformer.cuh"

void Transformer::PositionalEncoding(MatrixHalf *in, MatrixHalf *out) {
    //pe->print("pe:", true);
    in->addMatrix(pe, out);
}

void Transformer::make_pe(int batch, int max_len, int d_model, MatrixHalf *out) {
    auto div_term = new double[d_model / 2 + 1];
    for (int i = 0; i < d_model / 2 + 1; i++) {
        div_term[i] = exp(i * 2 * -log(10000.0) / d_model);
    }
    auto tmp_pe = new double[max_len * (d_model / 2 + 1)];
    for (int i = 0; i < max_len; i++) {
        for (int j = 0; j < d_model / 2 + 1; j++) {
            tmp_pe[i * (d_model / 2 + 1) + j] = (i + 1) * div_term[j];
        }
    }
    half *pe = new half[max_len * d_model];
    for (int i = 0; i < max_len; i++) {
        for (int j = 0; j < d_model; j++) {
            int idx = i * d_model + j;
            if (j % 2 == 0) {
                pe[idx] = sin(tmp_pe[i * (d_model / 2 + 1) + j / 2]);
            } else {
                pe[idx] = cos(tmp_pe[i * (d_model / 2 + 1) + j / 2]);
            }
        }
    }
    half *d_pe;
    cudaMalloc(&d_pe, sizeof(half) * batch * max_len * d_model);
    for (int i = 0; i < batch; i++) {
        cudaMemcpy(d_pe + i * max_len * d_model, pe, sizeof(half) * max_len * d_model, cudaMemcpyHostToDevice);
    }
    out->setMatrix(d_pe);

    delete[] div_term;
    delete[] tmp_pe;
    delete[] pe;
}

void Transformer::init(int batch, int max_len, int d_model) {

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

void Transformer::forward(int *en_in, int *de_in, MatrixHalf *out) {
    // 1. embedding
    auto ebd_t = new CudaTime();
    ebd_t->initAndStart();
    auto en_ebd_out = new MatrixHalf(batch, en_max_len, d_model, true);
    auto de_ebd_out = new MatrixHalf(batch, de_max_len, d_model, true);
    Embedding(en_in, batch, en_max_len, en_ebd_out);
    Embedding(de_in, batch, de_max_len, de_ebd_out);
    //printf("ebd time: %fms\n", ebd_t->endAndGetTime());

    // 2. position encoding
    auto pe_t = new CudaTime();
    pe_t->initAndStart();
    auto en_pe_out = new MatrixHalf(batch, en_max_len, d_model, true);
    auto de_pe_out = new MatrixHalf(batch, de_max_len, d_model, true);
    PositionalEncoding(en_ebd_out, en_pe_out);
    PositionalEncoding(de_ebd_out, de_pe_out);
    //printf("pe time: %fms\n", pe_t->endAndGetTime());

    // 3. encoder
    auto en_t = new CudaTime();
    en_t->initAndStart();
    auto encoder_out = new MatrixHalf(batch, en_max_len, d_model, true);
    encoder->forwardN(en_pe_out, encoder_out, 6);
    //printf("encoder time: %fms\n", en_t->endAndGetTime());

    // 4. decoder
    auto de_t = new CudaTime();
    de_t->initAndStart();
    auto decoder_out = new MatrixHalf(batch, de_max_len, d_model, true);
    decoder->forwardN(de_pe_out, encoder_out, decoder_out, 6);
    //printf("decoder time: %fms\n", de_t->endAndGetTime());

    // 5. liner + softmax
    auto lr_out = new MatrixHalf(batch, de_max_len, de_max_len, true);
    decoder_out->gemm_batches(W_last, lr_out, true);
    auto sm_out = new MatrixHalf(batch, de_max_len, de_max_len, true);
    softmax_cudnn_trans(lr_out->getMatrix(), batch * de_max_len, de_max_len, 1, 1, sm_out->getMatrix());


    sm_out->copyTo(out);

    // 6. free
    en_ebd_out->free_matrix();
    de_ebd_out->free_matrix();
    en_pe_out->free_matrix();
    de_pe_out->free_matrix();
    encoder_out->free_matrix();
    decoder_out->free_matrix();
    lr_out->free_matrix();
    sm_out->free_matrix();
}

Transformer::Transformer(const int batch, const int enMaxLen, const int deMaxLen, const int dModel) :
                                batch(batch), en_max_len(enMaxLen), de_max_len(deMaxLen), d_model(dModel) {
    int max_len = enMaxLen > deMaxLen ? enMaxLen : deMaxLen;
    pe = new MatrixHalf(batch, max_len, dModel, true);
    W_ebd = new MatrixHalf(1, max_len, d_model, true, "../../data/transformer/w_ebd");
    W_last = new MatrixHalf(1, d_model, max_len, true, "../../data/transformer/w_last");
    encoder = new Encoder();
    encoder->init(enMaxLen);
    decoder = new Decoder();
    decoder->init(deMaxLen);
}
