//
// Created by dbettkk on 2022/3/29.
//

#include "MatrixParam.cuh"


MatrixParam::MatrixParam(int m, int k, int n, half *matA, half *matB, half *matC, half *matD, half *cmprA,
                         int *binIndex) : m(m), k(k), n(n), matA(matA), matB(matB), matC(matC), matD(matD),
                                          cmprA(cmprA), binIndex(binIndex) {}

MatrixParam::MatrixParam(int m, int k, int n, half *matA, half *matB, half *matC, half *matD) : m(m), k(k), n(n),
                                                                                                matA(matA), matB(matB),
                                                                                                matC(matC),
                                                                                                matD(matD) {}

MatrixParam::MatrixParam(int m, int k, int n) : m(m), k(k), n(n), matA(nullptr), matB(nullptr), matC(nullptr),
                                                matD(nullptr),
                                                cmprA(nullptr), binIndex(nullptr) {}

MatrixParam::MatrixParam() = default;

void MatrixParam::printMatrix(half *item, int row, int col, const std::string& msg) {
    printf("%s\n", msg.c_str());
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", __half2int_rz(item[i * col + j]));
            //printf("%f ", __half2float(item[i * col + j]));
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
    half *cpu = new half[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum  = 0.0f;
            for (int v = 0; v < k; v++) {
                int posA =  i * k + v; // A[i][v]
                int posB =  v * n + j; // B[v][j]
                sum += __half2float(matA[posA]) * __half2float(matB[posB]);
            }
            int posRet = i * n + j;
            cpu[posRet] = __float2half(sum);  // [i][j]
        }
        // 检验速度用 printf("%d\n", i);
    }
    if (isPrintMatrix) printMatrix(cpu, m, n, "cpu");
    // diff
    int total = m * n, cnt = 0;
    int p = 0;
    for (int i = 0; i < m * n; i++) {
        if (abs(__half2float(matD[i]) - __half2float(cpu[i])) > 0.0001) {
            cnt++;
            if (p < 2) {
                printf("%f:%f\n", __half2float(matD[i]), __half2float(cpu[i]));
                p++;
            }
        } else {
            if (p < 1) {
                printf("%f:%f\n", __half2float(matD[i]), __half2float(cpu[i]));
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
    if (matA == nullptr) matA = new half[m * k];
    if (matB == nullptr) matB = new half[k * n];
    if (matC == nullptr) matC = new half[m * n];
    if (matD == nullptr) matD = new half[m * n];
}

void MatrixParam::readFromBin(const std::string& matAPath, const std::string& matBPath, const std::string& matCPath) {
    // float32的数据 需要转为float16
    std::ifstream inA(matAPath, std::ios::binary);
    std::ifstream inB(matBPath, std::ios::binary);
    std::ifstream inC(matCPath, std::ios::binary);

    initIfNull();

    inA.read((char *)matA, m * k * sizeof(half));
    inB.read((char *)matB, k * n * sizeof(half));
    inC.read((char *)matC, m * n * sizeof(half));

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

    for (int i = 0; i < m * k; i++)  matA[i] = __uint2half_rn(u(e));
    for (int i = 0; i < k * n; i++)  matB[i] = __uint2half_rn(u(e));
    for (int i = 0; i < m * n; i++)  matC[i] = __uint2half_rn(0);

}

void MatrixParam::generateRandSparseData(int bound) {
    // todo: 还未做
    initIfNull();
    // random
    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_int_distribution<unsigned> u(0, bound); // 闭区间

    for (int i = 0; i < m * k; i++)  matA[i] = __uint2half_rn(u(e));
    for (int i = 0; i < k * n; i++)  matB[i] = __uint2half_rn(u(e));
    for (int i = 0; i < m * n; i++)  matC[i] = __uint2half_rn(0);
}

half *MatrixParam::getMatD() const {
    return matD;
}

void MatrixParam::copyFromDevice(const half* dA, const half* dB, const half* dC, const half* dD, int inputM, int inputK, int inputN) {
    // 从gpu上拷贝矩阵数据
    initIfNull();
    restoreMatrix<half>(dA, inputM, inputK, matA, m, k, false);
    restoreMatrix<half>(dB, inputK, inputN, matB, k, n, false);
    restoreMatrix<half>(dC, inputM, inputN, matC, m, n, false);
    restoreMatrix<half>(dD, inputM, inputN, matD, m, n, false);
}

void MatrixParam::copyToDevice(half *dMatrix, char whichMatrix) {
    switch (whichMatrix) {
        case 'A': CHECK_CUDA( cudaMemcpy(dMatrix, matA, m * k * sizeof(half), cudaMemcpyHostToDevice) ) break;
        case 'B': CHECK_CUDA( cudaMemcpy(dMatrix, matB, k * n * sizeof(half), cudaMemcpyHostToDevice) ) break;
        case 'C': CHECK_CUDA( cudaMemcpy(dMatrix, matC, m * n * sizeof(half), cudaMemcpyHostToDevice) ) break;
        case 'D': CHECK_CUDA( cudaMemcpy(dMatrix, matD, m * n * sizeof(half), cudaMemcpyHostToDevice) ) break;
        default:
            printf("==[copyToDevice] nothing to copy==\n");
    }
}

half *MatrixParam::getMatA() const {
    return matA;
}

half *MatrixParam::getMatB() const {
    return matB;
}

void MatrixParam::setMatD(half *paramD) {
    MatrixParam::matD = paramD;
}

MatrixParam::~MatrixParam() {
    delete[] matA;
    delete[] matB;
    delete[] matC;
    delete[] matD;
    //printf("delete matrix\n");
}

