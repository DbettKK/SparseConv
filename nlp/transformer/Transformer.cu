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
    make_pe(max_len, d_model, pe);
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
