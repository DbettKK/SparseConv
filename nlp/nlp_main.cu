//
// Created by dbettkk on 2022/7/14.
//

#include "./transformer/Attention.cuh"
#include "../spmma/utils/CudaTime.cuh"
#include "transformer/Transformer.cuh"
#include "test/test_nlp.cuh"
#include <random>

float generate_random() {
    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_real_distribution<float> u(0, 1); // 闭区间
    return u(e);
}

void test_mask() {
    int sen = 16, ebd = 512, h = 8;
    half *att = new half[sen * sen];
    int *masks = new int[sen * sen];
    memset(masks, 0, sizeof(int) * sen * sen);
    half *V = new half[sen * ebd / h];
    for (int i = 0; i < sen * sen; i++) {
        att[i] = generate_random();
    }
    for (int i = 0; i < sen * ebd / h; i++) {
        V[i] = generate_random();
    }
    for (int i = 0; i < sen * sen; i+=3) masks[i] = 1;

    for (int i = 0; i < sen; i++) {
        for (int j = 0; j < sen; j++) {
            printf("%.2f ", __half2float(att[i * sen + j]));
        }
        printf("\n");
    }
    printf("masks: \n");
    for (int i = 0; i < sen; i++) {
        for (int j = 0; j < sen; j++) {
            printf("%d ", masks[i * sen + j]);
        }
        printf("\n");
    }

    half *dAtt, *dV;
    int *dM;
    cudaMalloc(&dAtt, sizeof(half) * sen * sen);
    cudaMalloc(&dV, sizeof(half) * sen * ebd / h);
    cudaMalloc(&dM, sizeof(int) * sen * sen);

    cudaMemcpy(dAtt, att, sizeof(half) * sen * sen, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, sizeof(half) * sen * ebd / h, cudaMemcpyHostToDevice);
    cudaMemcpy(dM, masks, sizeof(int) * sen * sen, cudaMemcpyHostToDevice);

    bool doMask = true;

    half *out = new half[sen * ebd / h];
    for (int i = 0; i < 10; i++) {
        auto t = new CudaTime();
        t->initAndStart();

        if (doMask) {
            mask_matrix_gpu<<<16, 32>>>(dAtt, dM, sen, sen);
            //sparse_mma_gemm_device(dAtt, dV, sen, sen, ebd / h, true, out);
            cusparse_gemm_csr_device(dAtt, dV, sen, sen, ebd / h, out);
        } else {
            cublas_gemm_device(dAtt, dV, sen, sen, ebd / h, out);
        }

        float time = t->endAndGetTime();
        printf("time: %fms\n", time);
    }

    for (int i = 0; i < sen; i++) {
        for (int j = 0; j < ebd / h; j++) {
            printf("%.2f ", __half2float(out[i * ebd / h + j]));
        }
    }
}

void test_gemm() {
    auto A = new MatrixHalf(1, 16, 512, true, 1);
    auto B = new MatrixHalf(1, 512, 512, true, 1);
    auto ans = new MatrixHalf(1, 16, 512, true);
    auto outA = new MatrixHalf(1, 16 * 8, 64, true);
    A->gemm(B, ans);
    ans->reshape(outA, 8);
    ans->print("ans: ", true);
    outA->print("outA: ", true);
}

void test_trans() {
    auto t = new Transformer(2, 16, 1, 512);
    int *en_in = new int[2 * 16];
    int *de_in = new int[2 * 1];
    for (int i = 0; i < 2 * 16; i++) en_in[i] = i / 2 + 1;
    for (int i = 0; i < 2 * 1; i++) de_in[i] = i / 2 + 1;
    auto out = new MatrixHalf(2, 1, 20, true);
    for (int i = 0; i < 12; i++) {
        auto trans_t = new CudaTime();
        trans_t->initAndStart();
        t->forward(en_in, de_in, out);
        out->print("out", true);
        printf("trans time: %fms\n", trans_t->endAndGetTime());
    }
}

int main() {
    test_gemm_batches();
    //test_trans();
    return 0;
}