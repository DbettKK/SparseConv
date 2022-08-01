//
// Created by dbettkk on 2022/8/1.
//

#ifndef SPARSECONV_MYRESNET_CUH
#define SPARSECONV_MYRESNET_CUH

#include "../interface/resnet_kernel.cuh"
#include "entity/MyTensor.cuh"

class MyResNet {

    void resnet50();

    void bottleneck1(MyTensor *input, MyTensor *output, int times);

    void bottleneck2(MyTensor *input, MyTensor *output, int times);

    void bottleneck3(MyTensor *input, MyTensor *output, int times);

    void bottleneck4(MyTensor *input, MyTensor *output, int times);
};


#endif //SPARSECONV_MYRESNET_CUH
