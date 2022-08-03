//
// Created by dbettkk on 2022/8/1.
//

#ifndef SPARSECONV_MYRESNET_CUH
#define SPARSECONV_MYRESNET_CUH

#include "../interface/resnet_kernel.cuh"
#include "entity/MyTensor.cuh"

class MyResNet {
    MyTensor *W_fc = new MyTensor(1, 1, 2048, 1000, true, 1);

    static void bottleneck1(MyTensor *input, MyTensor *output, int times);

    static void bottleneck2(MyTensor *input, MyTensor *output, int times);

    static void bottleneck3(MyTensor *input, MyTensor *output, int times);

    static void bottleneck4(MyTensor *input, MyTensor *output, int times);

public:
    void resnet50();
};


#endif //SPARSECONV_MYRESNET_CUH
