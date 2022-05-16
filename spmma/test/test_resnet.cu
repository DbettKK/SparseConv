//
// Created by dbettkk on 2022/5/14.
// test resnet-50
//
#include"../kernels/sparse_conv.cuh"

void test_conv(int d_n, int d_c, int d_h, int d_w, int f_n, int f_h, int f_w, int padding, int stride,
               const std::string &d_path,
               const std::string &f_path) {
    int f_c = d_c;
    // conv
    auto data = new Tensor4d(d_n, d_c, d_h, d_w);
    data->generateRandData(10);
    //data->readFromBin(d_path);
    auto filter = new Tensor4d(f_n, f_c, f_h, f_w);
    filter->generateRandSpData(10);
    //filter->readFromBin(f_path);
    auto param = new ConvParam(data, filter, padding, stride, 1);
    auto ret = sparse_conv(param);
    //printf("n:%d, c:%d, h: %d, w:%d\n", ret->getN(), ret->getC(), ret->getH(), ret->getW());
    delete data;
    delete filter;
    delete ret;
}

void test_resnet50() {
    // conv_1
    test_conv(1,3,224,224,64,7,7,3,2,"","");
    // conv_2
    test_conv(1,64,56,56,64,1,1,0,1,"","");
    test_conv(1,64,56,56,64,3,3,1,1,"","");
    test_conv(1,64,56,56,256,1,1,0,1,"","");
    for (int i = 0; i < 2; i++) {
        test_conv(1,256,56,56,64,1,1,0,1,"","");
        test_conv(1,64,56,56,64,3,3,1,1,"","");
        test_conv(1,64,56,56,256,1,1,0,1,"","");
    }
    // conv_3
    test_conv(1,256,56,56,128,1,1,0,1,"","");
    test_conv(1,128,28,28,128,3,3,1,1,"","");
    test_conv(1,128,28,28,512,1,1,0,1,"","");
    for (int i = 0; i < 3; i++) {
        test_conv(1,512,28,28,128,1,1,0,1,"","");
        test_conv(1,128,28,28,128,3,3,1,1,"","");
        test_conv(1,128,28,28,512,1,1,0,1,"","");
    }
    //conv_4
    test_conv(1,512,28,28,256,1,1,0,1,"","");
    test_conv(1,256,14,14,256,3,3,1,1,"","");
    test_conv(1,256,14,14,1024,1,1,0,1,"","");
    for (int i = 0; i < 5; i++) {
        test_conv(1,1024,14,14,256,1,1,0,1,"","");
        test_conv(1,256,14,14,256,3,3,1,1,"","");
        test_conv(1,256,14,14,1024,1,1,0,1,"","");
    }
    // conv_5
    test_conv(1,1024,14,14,512,1,1,0,1,"","");
    test_conv(1,512,7,7,512,3,3,1,1,"","");
    test_conv(1,512,7,7,2048,1,1,0,1,"","");
    for (int i = 0; i < 2; i++) {
        test_conv(1,2048,7,7,512,1,1,0,1,"","");
        test_conv(1,512,7,7,512,3,3,1,1,"","");
        test_conv(1,512,7,7,2048,1,1,0,1,"","");
    }
}

int main() {
    test_resnet50();
}
