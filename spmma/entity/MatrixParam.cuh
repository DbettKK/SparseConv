//
// Created by dbettkk on 2022/3/29.
//

#ifndef SPARSECONVOLUTION_MATRIXPARAM_CUH
#define SPARSECONVOLUTION_MATRIXPARAM_CUH

#include<iostream>
#include<fstream>
#include<random>
#include<cuda_fp16.h>
#include "../utils/util.cuh"

class MatrixParam {
    int m{}, k{}, n{};
    float *matA{}, *matB{}, *matC{}, *matD{};

public:
    MatrixParam(int m, int k, int n, float *matA, float *matB, float *matC, float *matD);

    MatrixParam(int m, int k, int n);

    MatrixParam();

    ~MatrixParam();

    float *getMatA() const;

    float *getMatB() const;

    float *getMatD() const;

    void setMatD(float *matD);

    static void printMatrix(float *item, int row, int col, const std::string& msg);

    void printMatrix(char whichMatrix);

    void printAllMatrix();

    bool checkCorrect(bool isPrintMatrix);

    void initIfNull();

    void readFromBin(const std::string& matAPath, const std::string& matBPath, const std::string& matCPath);

    void generateRandData(int bound);

    void generateRandSparseData(int bound);

    void copyFromDevice(const float* dA, const float* dB, const float* dC, const float* dD, int inputM, int inputK, int inputN);

    void copyToDevice(float *dMatrix, char whichMatrix);
};


#endif //SPARSECONVOLUTION_MATRIXPARAM_CUH
