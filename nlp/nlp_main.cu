//
// Created by dbettkk on 2022/7/14.
//

#include "./transformer/Attention.cuh"
#include "transformer/Transformer.cuh"
#include <random>


void test_trans() {
    int batch = 2, sen = 512;
    auto t = new Transformer(batch, sen, sen, 512, 37000);
    int *en_in = new int[batch * sen];
    int *de_in = new int[batch * sen];
    for (int i = 0; i < batch * sen; i++) en_in[i] = i / 2 + 1;
    for (int i = 0; i < batch * sen; i++) de_in[i] = i / 2 + 1;
    auto out = new MatrixHalf(batch, sen, 37000, true);
    for (int i = 0; i < 2; i++) {
        auto trans_t = new CudaTime();
        trans_t->initAndStart();
        t->forward(en_in, de_in, out);
        //out->print("out", true);
        printf("trans time: %fms\n", trans_t->endAndGetTime());
    }
}

int main() {
    test_trans();
    return 0;
}