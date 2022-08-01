//
// Created by dbettkk on 2022/8/1.
//

#ifndef SPARSECONV_MYRESNET_CUH
#define SPARSECONV_MYRESNET_CUH

#include "../interface/resnet_kernel.cuh"
#include "entity/MyTensor.cuh"

class MyResNet {

    void resnet50();

    void bottleneck(int in_c, int out_c, int stride);
};


#endif //SPARSECONV_MYRESNET_CUH
