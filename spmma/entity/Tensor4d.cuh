//
// Created by dbettkk on 2022/3/29.
//

#ifndef SPARSECONVOLUTION_TENSOR4D_CUH
#define SPARSECONVOLUTION_TENSOR4D_CUH

#include<iostream>
#include<fstream>
#include<random>
#include <cuda_fp16.h>

class Tensor4d {
    float *tensor;
    int n, c, h, w;

public:
    Tensor4d();

    Tensor4d(int n, int c, int h, int w);

    Tensor4d(float *tensor, int n, int c, int h, int w);

    ~Tensor4d();

    float *getTensor() const;

    int getN() const;

    int getC() const;

    int getH() const;

    int getW() const;

    int getTotalSize() const;

    void printTensor(const std::string& msg);

    void readFromBin(const std::string& path);

    void generateRandData(int bound);

    void generateRandSpData(int bound);
};


#endif //SPARSECONVOLUTION_TENSOR4D_CUH
