//
// Created by dbettkk on 2022/7/14.
//

#include "interface/sparse_mma_gemm.cuh"

//void transfomer() {
//    const int batch = 1, max_sen_len = 8, ebd = 512;
//    const int layers = 6, head = 8, n_ff = 2048;
//    /* encoder */
//    // 省略embedding层
//    half *input = new half[batch * max_sen_len * ebd];
//    // position encoding位置编码
//    position_encoding(input, batch, max_sen_len, ebd);
//    // core
//    half *Q = new half[ebd * ebd];
//    half *K = new half[ebd * ebd];
//    half *V = new half[ebd * ebd];
//    half *Wq = new half[ebd * ebd];
//    half *Wk = new half[ebd * ebd];
//    half *Wv = new half[ebd * ebd];
//    for (int layer = 0; layer < layers; layer++) {
//        // attention层
//        for (int i = 0; i < head; i++) {
//            // 得到当前头的 Q K V
//            sparse_mma_gemm_device(input, Wq + i * ebd * ebd / head, max_sen_len, ebd, ebd / head, true, Q + i * ebd * ebd / head);
//            sparse_mma_gemm_device(input, Wk + i * ebd * ebd / head, max_sen_len, ebd, ebd / head, true, K + i * ebd * ebd / head);
//            sparse_mma_gemm_device(input, Wv + i * ebd * ebd / head, max_sen_len, ebd, ebd / head, true, V + i * ebd * ebd / head);
//            // softmax transpose
//            half *K_transpose = new half[ebd * ebd];
//            transpose<<<1, 1>>>(Wk, K_transpose, 1, 1);
//            half *tmp = new half[1];
//            sparse_mma_gemm_device(Q + i * ebd * ebd / head, K_transpose + i * ebd * ebd / head,
//                                   max_sen_len, ebd / head, max_sen_len, true, tmp);
//            softmax_s(tmp);
//            // 和V相乘得到结果
//            half *ans = new half[1];
//            sparse_mma_gemm_device(tmp, V + i * ebd * ebd / head, max_sen_len, max_sen_len, ebd / head, true, ans);
//        }
//        // liner
//
//        // mlp层
//        half *W_ff = new half[ebd * n_ff];
//        half *W_ff_re = new half[ebd * n_ff];
//
//    }
//
//    /* decoder */
//
//}
//

//
//
//void test_cublas() {
//    half *hA = new half[256];
//    half *hB = new half[256];
//    half *hC = new half[256];
//    for (int i = 0; i < 256; i+=2) {
//        hA[i] = 1;
//        hA[i + 1] = 0;
//    }
//    for (int i = 0; i < 256; i++) hB[i] = 2;
//
//    half *dA;
//    int *dM;
//    cudaMalloc(&dA, sizeof(half) * 256);
//    cudaMalloc(&dM, sizeof(int) * 256);
//    cudaMemcpy(dA, hA, sizeof(half) * 256, cudaMemcpyHostToDevice);
//    cudaMemset(dM, 0, sizeof(int) * 256);
//
//    for (int i = 0; i < 10; i++) {
//        auto t = new CudaTime();
//        t->initAndStart();
//
//        cublas_gemm(hA, hB, 16, 16, 16, hC);
//        //sparse_mma_gemm_device(hA, hB, 16, 16, 16, true, hC);
//        //mask_matrix_gpu<<<1, 32>>>(dA, dM, 16, 16);
//
//        float total = t->endAndGetTime();
//        printf("time: %fms\n", total);
//    }
//
//
//    //for (int i = 0; i < 8; i++) printf("%d ", __half2int_rz(hC[i]));
//}
//
//void test_cusparse() {
//    half h_dense[]    = {0.0f,  0.0f,  1.0f,  2.0f,  0.0f,  0.0f,
//                          0.0f,  0.0f,  3.0f,  4.0f,  0.0f,  0.0f,
//                          5.0f,  6.0f,  0.0f,  0.0f,  7.0f,  8.0f,
//                          9.0f, 10.0f,  0.0f,  0.0f, 11.0f, 12.0f };
//    half *dA;
//    cudaMalloc(&dA, sizeof(half) * 24);
//    cudaMemcpy(dA, h_dense, sizeof(half) * 24, cudaMemcpyHostToDevice);
//
//    cusparse_blocked_ell_gemm_device(dA, nullptr, 4, 6, 1, nullptr);
//}
//
//void test_shared() {
//    half *item = new half[32 * 128];
//    for (int i = 0; i < 32*128; i++) item[i] = 1;
//    half *sum = new half[32 * 128];
//    half *di, *ds;
//    cudaMalloc(&di, sizeof(half) * 32 * 128);
//    cudaMalloc(&ds, sizeof(half) * 32 * 128);
//    cudaMemcpy(di, item, sizeof(half) * 32 * 128, cudaMemcpyHostToDevice);
//    cudaMemcpy(ds, sum, sizeof(half) * 32 * 128, cudaMemcpyHostToDevice);
//    softmax<<<128, 32>>>(di, ds, 32, 128);
//    cudaMemcpy(sum, ds, sizeof(half) * 32 * 128, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < 32 * 128; i++) printf("%f ", __half2float(sum[i]));
//}
//
//int main() {
//    test_shared();
//    return 0;
////    for (int i = 0; i < 10; i++) {
////        auto t = new CudaTime();
////        t->initAndStart();
////        cusparse_block_gemm();
////        printf("time: %fms\n", t->endAndGetTime());
////    }
////    return 0;
//    half *hA = new half[256];
//    half *hB = new half[256];
//    half *hOut = new half[256];
//
//    for (int i = 0; i < 256; i+=2) {
//        hA[i] = 2;
//        hA[i + 1] = 0;
//    }
//    for (int i = 0; i < 256; i++) hB[i] = 1;
//
//    sparse_mma_gemm_device(hA, hB, 16, 16, 16, true, hOut);
//
//    for (int i = 0; i < 16; i++) {
//        for (int j = 0; j < 16; j++) {
//            printf("%d ", __half2int_rz(hOut[i]));
//        }
//        printf("\n");
//    }
//    return 0;
//}

#include "./transformer/Attention.cuh"
#include "../spmma/utils/CudaTime.cuh"

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

    bool doMask = false;

    half *out = new half[sen * ebd / h];
    for (int i = 0; i < 10; i++) {
        auto t = new CudaTime();
        t->initAndStart();

        if (doMask) {
            mask_matrix_gpu<<<16, 32>>>(dAtt, dM, sen, sen);
            sparse_mma_gemm_device(dAtt, dV, sen, sen, ebd / h, true, out);
            //cusparse_gemm_csr_device(dAtt, dV, sen, sen, ebd / h, out);
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

int main() {

    test_mask();
    return 0;
    auto a = new Attention();
    a->initW();


    auto mh = new MatrixHalf(1, 16, 512, true, 1);
    //mh->print("input:", true);
    for (int i = 0; i < 12; i++) {
        //auto t = new CudaTime();
        //t->initAndStart();
        //a->forward(mh);
        //cusparse_gemm_blocked_device_test();
        cusparse_gemm_csr_device_test();
        //printf("time: %fms\n", t->endAndGetTime());
    }

}