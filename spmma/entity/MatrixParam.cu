//
// Created by dbettkk on 2022/3/29.
//

#include "MatrixParam.cuh"


MatrixParam::MatrixParam(int m, int k, int n, float *matA, float *matB, float *matC, float *matD) : m(m), k(k), n(n),
                                                                                                matA(matA), matB(matB),
                                                                                                matC(matC),
                                                                                                matD(matD) {}

MatrixParam::MatrixParam(int m, int k, int n) : m(m), k(k), n(n), matA(nullptr), matB(nullptr), matC(nullptr),
                                                matD(nullptr) {}

MatrixParam::MatrixParam() = default;

void MatrixParam::printMatrix(float *item, int row, int col, const std::string& msg) {
    printf("%s\n", msg.c_str());
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", item[i * col + j]);
        }
        printf("\n");
    }
}

void MatrixParam::printMatrix(char whichMatrix) {
    switch (whichMatrix) {
        case 'A':
            printMatrix(matA, m, k, "A:");
            break;
        case 'B':
            printMatrix(matB, k, n, "B:");
            break;
        case 'C':
            printMatrix(matC, m, n, "C:");
            break;
        case 'D':
            printMatrix(matD, m, n, "D:");
            break;
        default:
            printf("nothing to print\n");
    }
}

void MatrixParam::printAllMatrix() {
    printf("m: %d\nk: %d\nn: %d\n", m, k, n);
    printMatrix(matA, m, k, "A:");
    printMatrix(matB, k, n, "B:");
    printMatrix(matC, m, n, "C:");
    printMatrix(matD, m, n, "D:");
}

bool MatrixParam::checkCorrect(bool isPrintMatrix) {
    auto cpu = new float[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum  = 0.0f;
            for (int v = 0; v < k; v++) {
                int posA =  i * k + v; // A[i][v]
                int posB =  v * n + j; // B[v][j]
                sum += matA[posA] * matB[posB];
            }
            int posRet = i * n + j;
            cpu[posRet] = sum;  // [i][j]
        }
    }
    if (isPrintMatrix) printMatrix(cpu, m, n, "cpu");
    // diff
    int total = m * n, cnt = 0;
    int p = 0;
    for (int i = 0; i < m * n; i++) {
        if (abs(matD[i] - cpu[i]) > 0.01) {
            cnt++;
            if(p < 2) {
                printf("%f:%f\n", matD[i], cpu[i]);
                p++;
            }
        }
    }
    printf("total: %d\tdiff: %d\n", total, cnt);
    if (isPrintMatrix) printMatrix(matD, m, n, "gpu:");
    delete[] cpu;
    return cnt == 0;
}

void MatrixParam::initIfNull() {
    if (matA == nullptr) matA = new float[m * k];
    if (matB == nullptr) matB = new float[k * n];
    if (matC == nullptr) matC = new float[m * n];
    if (matD == nullptr) matD = new float[m * n];
}

void MatrixParam::readFromBin(const std::string& matAPath, const std::string& matBPath, const std::string& matCPath) {
    // float32的数据 需要转为float16
    std::ifstream inA(matAPath, std::ios::binary);
    std::ifstream inB(matBPath, std::ios::binary);
    std::ifstream inC(matCPath, std::ios::binary);

    initIfNull();

    inA.read((char *)matA, m * k * sizeof(float));
    inB.read((char *)matB, k * n * sizeof(float));
    inC.read((char *)matC, m * n * sizeof(float));

    inA.close();
    inB.close();
    inC.close();
}

void MatrixParam::generateRandData(int bound) {
    initIfNull();
    // random
    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_int_distribution<unsigned> u(0, bound); // 闭区间

    for (int i = 0; i < m * k; i++)  matA[i] = float(u(e));
    for (int i = 0; i < k * n; i++)  matB[i] = float(u(e));
    for (int i = 0; i < m * n; i++)  matC[i] = 0;

}

void MatrixParam::generateRandSparseData(int bound) {
    // todo: 还未做
    initIfNull();
    // random
    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_int_distribution<unsigned> u(0, bound); // 闭区间

    for (int i = 0; i < m * k; i++)  matA[i] = float(u(e));
    for (int i = 0; i < k * n; i++)  matB[i] = float(u(e));
    for (int i = 0; i < m * n; i++)  matC[i] = float(0);
}

float *MatrixParam::getMatD() const {
    return matD;
}

void MatrixParam::copyFromDevice(const float* dA, const float* dB, const float* dC, const float* dD, int inputM, int inputK, int inputN) {
    // 从gpu上拷贝矩阵数据
    initIfNull();
    restoreMatrix<float>(dA, inputM, inputK, matA, m, k, false);
    restoreMatrix<float>(dB, inputK, inputN, matB, k, n, false);
    restoreMatrix<float>(dC, inputM, inputN, matC, m, n, false);
    restoreMatrix<float>(dD, inputM, inputN, matD, m, n, false);
}

void MatrixParam::copyToDevice(float *dMatrix, char whichMatrix) {
    switch (whichMatrix) {
        case 'A': CHECK_CUDA( cudaMemcpy(dMatrix, matA, m * k * sizeof(float), cudaMemcpyHostToDevice) ) break;
        case 'B': CHECK_CUDA( cudaMemcpy(dMatrix, matB, k * n * sizeof(float), cudaMemcpyHostToDevice) ) break;
        case 'C': CHECK_CUDA( cudaMemcpy(dMatrix, matC, m * n * sizeof(float), cudaMemcpyHostToDevice) ) break;
        case 'D': CHECK_CUDA( cudaMemcpy(dMatrix, matD, m * n * sizeof(float), cudaMemcpyHostToDevice) ) break;
        default:
            printf("==[copyToDevice] nothing to copy==\n");
    }
}

float *MatrixParam::getMatA() const {
    return matA;
}

float *MatrixParam::getMatB() const {
    return matB;
}

void MatrixParam::setMatD(float *paramD) {
    MatrixParam::matD = paramD;
}

MatrixParam::~MatrixParam() {
    delete[] matA;
    delete[] matB;
    delete[] matC;
    delete[] matD;
    //printf("delete matrix\n");
}

