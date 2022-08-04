//
// Created by dbettkk on 2022/8/1.
//

#include "test/tests.cuh"

void test_resnet() {
    auto mr = new MyResNet();
    for (int i = 0; i < 12; i++) {
        auto tt = new CudaTime();
        tt->initAndStart();
        mr->resnet50();
        printf("total time: %fms\n", tt->endAndGetTime());
    }

}

int main() {
    test_resnet();
    return 0;
}