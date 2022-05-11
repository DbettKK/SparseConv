//
// Created by dbettkk on 2022/3/29.
//

#include "Tensor4d.cuh"

Tensor4d::Tensor4d() {}

Tensor4d::Tensor4d(int n, int c, int h, int w) : tensor(nullptr), n(n), c(c), h(h), w(w) {}

Tensor4d::Tensor4d(float *tensor, int n, int c, int h, int w) : tensor(tensor), n(n), c(c), h(h), w(w) {}

int Tensor4d::getTotalSize() const {
    return n * c * h * w;
}

float *Tensor4d::getTensor() const {
    return tensor;
}

int Tensor4d::getN() const {
    return n;
}

int Tensor4d::getC() const {
    return c;
}

int Tensor4d::getH() const {
    return h;
}

int Tensor4d::getW() const {
    return w;
}

void Tensor4d::printTensor(const std::string& msg) {
    printf("%s\n", msg.c_str());
    for (int i = 0; i < n; i++) {
        printf("n%d:\n", i);
        for (int j = 0; j < c; j++) {
            printf("c%d:\n", j);
            for (int k = 0; k < h; k++) {
                for (int v = 0; v < w; v++) {
                    printf("%f ", tensor[i * c * h * w + j * h * w + k * w + v]);
                }
                printf("\n");
            }
        }
    }
    printf("\n");
}

void Tensor4d::readFromBin(const std::string &path) {
    std::ifstream in(path, std::ios::binary);

    if (tensor == nullptr) tensor = new float[getTotalSize()];

    in.read((char *)tensor, getTotalSize() * sizeof(float));

    in.close();
}

void Tensor4d::generateRandData(int bound) {
    // random
    if (tensor == nullptr) tensor = new float[getTotalSize()];

    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_int_distribution<unsigned> u(0, bound); // 闭区间

    for (int i = 0; i < getTotalSize(); i++)  {
        tensor[i] = float(u(e));
    }

//    if (tensor == nullptr) tensor = new half[getTotalSize()];
//    for (int i = 0; i < getTotalSize(); i++)  {
//        tensor[i] = rand() % bound;
//    }

}

void Tensor4d::generateRandSpData(int bound) {
    // random
    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_int_distribution<unsigned> u(1, bound); // 闭区间

    if (tensor == nullptr) tensor = new float[getTotalSize()];
//    int row = n;
//    int col = h * w * c;
//    for (int i = 0; i < row; i++) {
//        for (int j = 0; j < col; j+=2) {
//            if (j * 2 + 1 >= col) {
//                tensor[i * col + j * 2] = 0;
//            } else {
//                int rand_index = u(e) % 2 == 0 ? 1 : 0;
//                tensor[i * col + j * 2 + rand_index] = 0;
//                tensor[i * col + j * 2 + 1 - rand_index] = float(u(e));
//            }
//        }
//    }
    for (int i = 0; i < getTotalSize(); i+=2)  {
        int rand = u(e);
        tensor[i] = 0;
        tensor[i + 1] = rand;
    }
}

Tensor4d::~Tensor4d() {
    delete[] tensor;
    //printf("delete tensor\n");
}




