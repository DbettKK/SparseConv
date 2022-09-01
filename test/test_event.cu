//
// Created by dbettkk on 2022/8/30.
//
#include "test_event.cuh"

int main() {
    auto t = new CudaTime();
    t->initAndStart();
    for (int j = 0; j < 1000; j++)
        for (int i = 0; i < 10000; i++) ;
    t->endAndPrintTime("cpu time:");
    return 0;
}
