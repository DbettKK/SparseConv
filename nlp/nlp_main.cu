//
// Created by dbettkk on 2022/7/14.
//

#include "interface/sparse_mma_gemm.cuh"

void transfomer() {
    const int batch = 1, max_sen_len = 8, ebd = 512;
    const int layers = 6, head = 8, n_ff = 2048;
    /* encoder */
    // 省略embedding层
    half *input = new half[batch * max_sen_len * ebd];
    // position encoding位置编码
    position_encoding(input, batch, max_sen_len, ebd);
    // core
    half *Q = new half[ebd * ebd];
    half *K = new half[ebd * ebd];
    half *V = new half[ebd * ebd];
    half *Wq = new half[ebd * ebd];
    half *Wk = new half[ebd * ebd];
    half *Wv = new half[ebd * ebd];
    for (int layer = 0; layer < layers; layer++) {
        // attention层
        for (int i = 0; i < head; i++) {
            // 得到当前头的 Q K V
            sparse_mma_gemm_device(input, Wq + i * ebd * ebd / head, max_sen_len, ebd, ebd / head, true, Q + i * ebd * ebd / head);
            sparse_mma_gemm_device(input, Wk + i * ebd * ebd / head, max_sen_len, ebd, ebd / head, true, K + i * ebd * ebd / head);
            sparse_mma_gemm_device(input, Wv + i * ebd * ebd / head, max_sen_len, ebd, ebd / head, true, V + i * ebd * ebd / head);
            // softmax transpose
            half *K_transpose = new half[ebd * ebd];
            transpose<<<1, 1>>>(Wk, K_transpose, 1, 1);
            half *tmp = new half[1];
            sparse_mma_gemm_device(Q + i * ebd * ebd / head, K_transpose + i * ebd * ebd / head,
                                   max_sen_len, ebd / head, max_sen_len, true, tmp);
            softmax(tmp);
            // 和V相乘得到结果
            half *ans = new half[1];
            sparse_mma_gemm_device(tmp, V + i * ebd * ebd / head, max_sen_len, max_sen_len, ebd / head, true, ans);
        }
        // liner

        // mlp层
        half *W_ff = new half[ebd * n_ff];
        half *W_ff_re = new half[ebd * n_ff];

    }

    /* decoder */

}

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
    printf("V: \n");
    for (int i = 0; i < sen; i++) {
        for (int j = 0; j < ebd / h; j++) {
            printf("%.2f ", __half2float(V[i * sen + j]));
        }
        printf("\n");
    }

    //定义两个cuda事件类型
    cudaEvent_t start, stop;

    //初始化
    //cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
    //cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //代表事件开始
    cudaEventRecord(start);
    cudaEventQuery(start);


    //代表事件结束
    cudaEventRecord(stop);
    //保证事件结束这一纪录完成后再执行后面的代码
    cudaEventSynchronize(stop);

    //计算start和stop之间的时间差
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("time: %fms\n", elapsed_time);

    //销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

void test_cublas() {
    half *hA = new half[256];
    half *hB = new half[256];
    half *hC = new half[256];
    for (int i = 0; i < 256; i+=2) {
        hA[i] = 1;
        hA[i + 1] = 0;
    }
    for (int i = 0; i < 256; i++) hB[i] = 2;

    half *dA;
    int *dM;
    cudaMalloc(&dA, sizeof(half) * 256);
    cudaMalloc(&dM, sizeof(int) * 256);
    cudaMemcpy(dA, hA, sizeof(half) * 256, cudaMemcpyHostToDevice);
    cudaMemset(dM, 0, sizeof(int) * 256);

    for (int i = 0; i < 10; i++) {
        auto t = new CudaTime();
        t->initAndStart();

        cublas_gemm(hA, hB, 16, 16, 16, hC);
        //sparse_mma_gemm_device(hA, hB, 16, 16, 16, true, hC);
        //mask_matrix_gpu<<<1, 32>>>(dA, dM, 16, 16);

        float total = t->endAndGetTime();
        printf("time: %fms\n", total);
    }


    //for (int i = 0; i < 8; i++) printf("%d ", __half2int_rz(hC[i]));
}

int main() {
    test_cublas();
    return 0;
    half *hA = new half[256];
    half *hB = new half[256];
    half *hOut = new half[256];

    for (int i = 0; i < 256; i+=2) {
        hA[i] = 2;
        hA[i + 1] = 0;
    }
    for (int i = 0; i < 256; i++) hB[i] = 1;

    sparse_mma_gemm_device(hA, hB, 16, 16, 16, true, hOut);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%d ", __half2int_rz(hOut[i]));
        }
        printf("\n");
    }
    return 0;
}