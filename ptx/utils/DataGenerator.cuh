//
// Created by dbettkk on 2022/6/17.
//

#ifndef SPARSECONV_DATAGENERATOR_CUH
#define SPARSECONV_DATAGENERATOR_CUH

#include <cuda_fp16.h>
#include <random>
#include <ctime>

class DataGenerator {
    int generateRandomNumber(int upBound, int downBound);
public:
    half *generateZero(int row, int col);

    half *generateNumber(int row, int col, int number);

    half *generateRandom(int row, int col);

    half *generateSimpleSparse(int row, int col, bool isStructured);

    half *generateRandomSparse(int row, int col, bool isStructured);

    void printMatrix(half *item, int row, int col);
};


#endif //SPARSECONV_DATAGENERATOR_CUH
