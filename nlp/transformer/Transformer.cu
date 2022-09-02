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
            //printf("%.2f ", __half2float(pe[i * d_model + j]));
        }
        //printf("\n");
    }
    half *d_pe;
    CHECK_CUDA(cudaMalloc(&d_pe, sizeof(half) * batch * max_len * d_model));
    for (int i = 0; i < batch; i++) {
        CHECK_CUDA(cudaMemcpy(d_pe + i * max_len * d_model, pe, sizeof(half) * max_len * d_model, cudaMemcpyHostToDevice));
    }
    out->setMatrix(d_pe);

    delete[] div_term;
    delete[] tmp_pe;
    delete[] pe;
}


void Transformer::make_mask_spmma(int row, int col, int *out) {
    // 从主对角线开始 隔两个对角线的值不mask
    int *h_mask = new int[row * col];
    memset(h_mask, 0, sizeof(int) * row * col);
    int max_num = (col - 1) / 3;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            for (int k = 0; k <= max_num; k++) {
                if (i == j + k * 3) h_mask[i * col + j] = 1;
                if (j == i + k * 3) h_mask[i * col + j] = 1;
            }
        }
    }
    cudaMemcpy(out, h_mask, sizeof(int) * row * col, cudaMemcpyHostToDevice);
}

void Transformer::make_mask_src(int row, int col, int *out) {
    // 从主对角线开始 隔两个对角线的值不mask
    int *h_mask = new int[row * col];
    memset(h_mask, 0, sizeof(int) * row * col);
    int max_num = (col - 1) / 3;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            for (int k = 0; k <= max_num; k++) {
                if (i == j + k * 3) h_mask[i * col + j] = 1;
                if (j == i + k * 3) h_mask[i * col + j] = 1;
            }
        }
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (j > i) h_mask[i * col + j] = 0;
        }
    }
    cudaMemcpy(out, h_mask, sizeof(int) * row * col, cudaMemcpyHostToDevice);

}

void Transformer::Embedding(const int *in, int max_len, MatrixHalf *out) {
    // in: [batch, max_len] type: INT/LONG  HOST
    // out: [batch, max_len, embedding_dim]
    int ebd_dim = out->getCol();
    half *one_hot = new half[batch * max_len * source_vocab];
    memset(one_hot, 0, sizeof(half) * batch * max_len * source_vocab);
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < max_len; j++) {
            // 即输入可以为[0, source_vocab]的任意数字 然后将其转换为向量
            int idx = i * max_len * source_vocab + j * source_vocab + in[i * max_len + j];
            one_hot[idx - 1] = 1;
        }
    }
    auto dh = new MatrixHalf(batch, max_len, source_vocab, true);
    cudaMemcpy(dh->getMatrix(), one_hot, sizeof(half) * batch * max_len * source_vocab, cudaMemcpyHostToDevice);

    dh->gemm_batches(W_ebd, out, true);

    delete[] one_hot;
    dh->free_matrix();
    //delete[] one_hot;
}

void Transformer::forward(const int *en_in, const int *de_in, MatrixHalf *out) {
    // encoder
    // 1. embedding  x Webd
    auto en_ebd_out = new MatrixHalf(batch, en_max_len, d_model, true);
    Embedding(en_in, en_max_len, en_ebd_out);
    //en_ebd_out->print("ebd", true);
    // 2. position encoding  + W_pe
    auto en_pe_out = new MatrixHalf(batch, en_max_len, d_model, true);
    PositionalEncoding(en_ebd_out, en_pe_out);
    //en_pe_out->print("pe:", true);
    // 3. self-attention + mlp  xN
    auto encoder_out = new MatrixHalf(batch, en_max_len, d_model, true);
    auto enc_t = new CudaTime();
    enc_t->initAndStart();
    encoder->forwardN(en_pe_out, encoder_out, 6);
    enc_t->endAndPrintTime("encoder:");
    //encoder_out->print("encoder", true);
    // decoder
    // 1. embedding
    auto de_ebd_out = new MatrixHalf(batch, de_max_len, d_model, true);
    Embedding(de_in, de_max_len, de_ebd_out);
    // 2. position encoding
    auto de_pe_out = new MatrixHalf(batch, de_max_len, d_model, true);
    PositionalEncoding(de_ebd_out, de_pe_out);
    // 3.  self-attention + attention + mlp
    auto de_t = new CudaTime();
    de_t->initAndStart();
    auto decoder_out = new MatrixHalf(batch, de_max_len, d_model, true);
    decoder->forwardN(de_pe_out, encoder_out, decoder_out, 6);
    //decoder_out->print("decoder", true);
    // generator
    // 1. liner
    auto lr_out = new MatrixHalf(batch, de_max_len, target_vocab, true);
    decoder_out->gemm_batches(W_last, lr_out, true);
    // 2. softmax
    auto sm_out = new MatrixHalf(batch, de_max_len, target_vocab, true);
    softmax_cudnn_trans(lr_out->getMatrix(), batch * de_max_len, target_vocab, 1, 1, sm_out->getMatrix());

    // output
    sm_out->copyTo(out);

    // free
    en_ebd_out->free_matrix();
    de_ebd_out->free_matrix();
    en_pe_out->free_matrix();
    de_pe_out->free_matrix();
    encoder_out->free_matrix();
    decoder_out->free_matrix();
    lr_out->free_matrix();
    sm_out->free_matrix();
}

Transformer::Transformer(const int batch, const int enMaxLen, const int deMaxLen, const int dModel, const int vocab) :
                                batch(batch), en_max_len(enMaxLen), de_max_len(deMaxLen), d_model(dModel), source_vocab(vocab), target_vocab(vocab) {
    int max_len = enMaxLen > deMaxLen ? enMaxLen : deMaxLen;
    pe = new MatrixHalf(batch, max_len, dModel, true);
    make_pe(batch, max_len, dModel, pe);
    //pe->print("pe:", true);
    W_ebd = new MatrixHalf(1, source_vocab, d_model, true, "../../data/transformer/w_ebd");
    W_last = new MatrixHalf(1, d_model, target_vocab, true, "../../data/transformer/w_last");
    // mask
    int *mask_en_spmma, *mask_de_spmma, *mask_de_src;
    CHECK_CUDA(cudaMalloc(&mask_en_spmma, sizeof(int) * enMaxLen * enMaxLen))
    CHECK_CUDA(cudaMalloc(&mask_de_spmma, sizeof(int) * deMaxLen * enMaxLen))
    CHECK_CUDA(cudaMalloc(&mask_de_src, sizeof(int) * deMaxLen * enMaxLen))
    make_mask_spmma(enMaxLen, enMaxLen, mask_en_spmma);
    make_mask_spmma(deMaxLen, enMaxLen, mask_de_spmma);
    make_mask_src(deMaxLen, enMaxLen, mask_de_src);

    encoder = new Encoder(mask_en_spmma);
    decoder = new Decoder(mask_de_spmma, mask_de_src);
}
