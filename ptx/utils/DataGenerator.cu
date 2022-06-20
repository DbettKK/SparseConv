//
// Created by dbettkk on 2022/6/17.
//

#include "DataGenerator.cuh"

int DataGenerator::generateRandomNumber(int upBound, int downBound) {
    std::random_device sd;
    std::default_random_engine e(sd());
    std::uniform_int_distribution<int> u(upBound, downBound); // 闭区间
    return u(e);
}

half *DataGenerator::generateZero(int row, int col) {
    return generateNumber(row, col, 0);
}

half *DataGenerator::generateNumber(int row, int col, int number) {
    half *ret = new half[row * col];
    for (int i = 0; i < row * col; i++) ret[i] = number;
    return ret;
}

half *DataGenerator::generateRandom(int row, int col) {
    half *ret = new half[row * col];
    for (int i = 0; i < row * col; i++) ret[i] = generateRandomNumber(0, 10);
    return ret;
}

half *DataGenerator::generateSimpleSparse(int row, int col, bool isStructured) {
    half *ret = new half[row * col];
    for (int i = 0; i < row * col; i+=4) {
//        int zero_0 = u(e);
//        int zero_1 = u(e);
        ret[i] = 0;
        ret[i + 1] = 1;
        ret[i + 2] = 0;
        ret[i + 3] = 1;
    }
    return ret;
}

half *DataGenerator::generateRandomSparse(int row, int col, bool isStructured) {
    half *ret = new half[row * col];
    for (int i = 0; i < row * col; i+=4) {
        int zero_0 = generateRandomNumber(0, 1);
        int zero_1 = generateRandomNumber(2, 3);
        ret[i + zero_0] = 0;
        ret[i + 1 - zero_0] = generateRandomNumber(0, 10);
        ret[i + zero_1] = 0;
        ret[i + 3 - zero_1] = generateRandomNumber(0, 10);
    }
    return ret;
}

void DataGenerator::printMatrix(half *item, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", __half2int_rz(item[i * col + j]));
        }
        printf("\n");
    }
}
