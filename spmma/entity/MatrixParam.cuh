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
    half *matA{}, *matB{}, *matC{}, *matD{};
    half *cmprA{};
    int *binIndex{};
public:
    MatrixParam(int m, int k, int n, half *matA, half *matB, half *matC, half *matD, half *cmprA, int *binIndex);

    MatrixParam(int m, int k, int n, half *matA, half *matB, half *matC, half *matD);

    MatrixParam(int m, int k, int n);

    MatrixParam();

    ~MatrixParam();

    half *getMatA() const;

    half *getMatB() const;

    half *getMatD() const;

    void setMatD(half *matD);

    static void printMatrix(half *item, int row, int col, const std::string& msg);

    void printMatrix(char whichMatrix);

    void printAllMatrix();

    bool checkCorrect(bool isPrintMatrix);

    void initIfNull();

    void readFromBin(const std::string& matAPath, const std::string& matBPath, const std::string& matCPath);

    void generateRandData(int bound);

    void generateRandSparseData(int bound);

    void copyFromDevice(const half* dA, const half* dB, const half* dC, const half* dD, int inputM, int inputK, int inputN);

    void copyToDevice(half *dMatrix, char whichMatrix);
};


#endif //SPARSECONVOLUTION_MATRIXPARAM_CUH
