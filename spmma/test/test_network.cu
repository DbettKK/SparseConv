//
// Created by dbettkk on 2022/4/17.
//
//
// Created by dbettkk on 2022/4/17.
//
#include"../kernels/sparse_conv.cuh"

half *read_bin(int size, const std::string& path) {
    half *ret = new half[size];
    std::ifstream in(path, std::ios::binary);
    in.read((char *)ret, sizeof(half) * size);
    //for(int i = 0; i < size; i++) ret[i] = int(ret[i]);
    in.close();
    return ret;
}

void test_conv(int d_n, int d_c, int d_h, int d_w, int f_n, int f_c, int f_h, int f_w, int padding, int stride,
               const std::string &d_path,
               const std::string &f_path) {
    // conv
    auto data = new Tensor4d(d_n, d_c, d_h, d_w);
    data->readFromBin(d_path);
    auto filter = new Tensor4d(f_n, f_c, f_h, f_w);
    filter->readFromBin(f_path);
    auto param = new ConvParam(data, filter, padding, stride, 1);
    CudaTime *t = new CudaTime();
    t->initAndStart();
    auto ret = sparse_conv(param);
    float tt = t->endAndGetTime();
    printf("tt: %f\n", tt);
    //printf("n:%d, c:%d, h: %d, w:%d\n", ret->getN(), ret->getC(), ret->getH(), ret->getW());
    delete data;
    delete filter;
    delete ret;
}

void test_matmul_conv(int d_n, int d_c, int d_h, int d_w, int f_n, int f_c, int f_h, int f_w, int padding, int stride,
                      const std::string &a_path,
                      const std::string &b_path) {
    int m = f_n;
    int k = f_c * f_h * f_w;
    int out_h = (d_h + 2 * padding - f_h) / stride + 1;
    int out_w = (d_w + 2 * padding - f_w) / stride + 1;
    int n = d_n * out_h * out_w;

    half *inA, *inB;
    half *hA = read_bin(m * k, a_path);
    half *hB = read_bin(k * n, b_path);
    CHECK_CUDA( cudaMalloc((void **)&inA, sizeof(half) * m * k) )
    CHECK_CUDA( cudaMalloc((void **)&inB, sizeof(half) * k * n) )
    CHECK_CUDA( cudaMemcpy(inA, hA, sizeof(half) * m * k, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(inB, hB, sizeof(half) * k * n, cudaMemcpyHostToDevice) )

    half *outputD;
    CHECK_CUDA( cudaMalloc((void **)&outputD, sizeof(half) * m * n) )
    auto test = new MatrixParam(m, k, n);
    spmma_matmul(inA, inB, m, k, n, false, outputD, test);

    test->checkCorrect(false);

    CHECK_CUDA( cudaFree(inA) )
    CHECK_CUDA( cudaFree(inB) )
    CHECK_CUDA( cudaFree(outputD) )
}

void test_vgg16() {
    std::string pre_path = "../data/vgg";
    test_conv(1, 3, 224, 224, 64, 3, 3, 3, 1, 1, pre_path + "/data1.bin", pre_path + "/filter1.bin");
    test_conv(1, 64, 224, 224, 64, 64, 3, 3, 1, 1, pre_path + "/data2.bin", pre_path + "/filter2.bin");
    test_conv(1, 64, 112, 112, 128, 64, 3, 3, 1, 1, pre_path + "/data3.bin", pre_path + "/filter3.bin");
    test_conv(1, 128, 112, 112, 128, 128, 3, 3, 1, 1, pre_path + "/data4.bin", pre_path + "/filter4.bin");
    test_conv(1, 128, 56, 56, 256, 128, 3, 3, 1, 1, pre_path + "/data5.bin", pre_path + "/filter5.bin");
    test_conv(1, 256, 56, 56, 256, 256, 3, 3, 1, 1, pre_path + "/data6.bin", pre_path + "/filter6.bin");
    test_conv(1, 256, 56, 56, 256, 256, 3, 3, 1, 1, pre_path + "/data7.bin", pre_path + "/filter7.bin");
    test_conv(1, 256, 28, 28, 512, 256, 3, 3, 1, 1, pre_path + "/data8.bin", pre_path + "/filter8.bin");
    test_conv(1, 512, 28, 28, 512, 512, 3, 3, 1, 1, pre_path + "/data9.bin", pre_path + "/filter9.bin");
    test_conv(1, 512, 28, 28, 512, 512, 3, 3, 1, 1, pre_path + "/data10.bin", pre_path + "/filter10.bin");
    test_conv(1, 512, 14, 14, 512, 512, 3, 3, 1, 1, pre_path + "/data11.bin", pre_path + "/filter11.bin");
    test_conv(1, 512, 14, 14, 512, 512, 3, 3, 1, 1, pre_path + "/data12.bin", pre_path + "/filter12.bin");
    test_conv(1, 512, 14, 14, 512, 512, 3, 3, 1, 1, pre_path + "/data13.bin", pre_path + "/filter13.bin");
}

void test_resnet18() {
    std::string pre_path = "../../data/resnet";
    test_conv(16, 3, 224, 224, 64, 3, 7, 7, 3, 2, pre_path + "/data1.bin", pre_path + "/filter1.bin");
    test_conv(16, 64, 56, 56, 64, 64, 3, 3, 1, 1, pre_path + "/data2.bin", pre_path + "/filter2.bin");
    test_conv(16, 64, 56, 56, 64, 64, 3, 3, 1, 1, pre_path + "/data3.bin", pre_path + "/filter3.bin");
    test_conv(16, 64, 56, 56, 64, 64, 3, 3, 1, 1, pre_path + "/data4.bin", pre_path + "/filter4.bin");
    test_conv(16, 64, 56, 56, 64, 64, 3, 3, 1, 1, pre_path + "/data5.bin", pre_path + "/filter5.bin");
    test_conv(16, 64, 56, 56, 128, 64, 3, 3, 1, 2, pre_path + "/data6.bin", pre_path + "/filter6.bin");
    test_conv(16, 128, 28, 28, 128, 128, 3, 3, 1, 1, pre_path + "/data7.bin", pre_path + "/filter7.bin");
    test_conv(16, 128, 28, 28, 128, 128, 3, 3, 1, 1, pre_path + "/data8.bin", pre_path + "/filter8.bin");
    test_conv(16, 128, 28, 28, 128, 128, 3, 3, 1, 1, pre_path + "/data9.bin", pre_path + "/filter9.bin");
    test_conv(16, 128, 28, 28, 256, 128, 3, 3, 1, 2, pre_path + "/data10.bin", pre_path + "/filter10.bin");
    test_conv(16, 256, 14, 14, 256, 256, 3, 3, 1, 1, pre_path + "/data11.bin", pre_path + "/filter11.bin");
    test_conv(16, 256, 14, 14, 256, 256, 3, 3, 1, 1, pre_path + "/data12.bin", pre_path + "/filter12.bin");
    test_conv(16, 256, 14, 14, 256, 256, 3, 3, 1, 1, pre_path + "/data13.bin", pre_path + "/filter13.bin");
    test_conv(16, 256, 14, 14, 512, 256, 3, 3, 1, 2, pre_path + "/data14.bin", pre_path + "/filter14.bin");
    test_conv(16, 512, 7, 7, 512, 512, 3, 3, 1, 1, pre_path + "/data15.bin", pre_path + "/filter15.bin");
    test_conv(16, 512, 7, 7, 512, 512, 3, 3, 1, 1, pre_path + "/data16.bin", pre_path + "/filter16.bin");
    test_conv(16, 512, 7, 7, 512, 512, 3, 3, 1, 1, pre_path + "/data17.bin", pre_path + "/filter17.bin");
}

void test_alexnet() {
    std::string pre_path = "../data/alex";
    test_conv(1, 3, 227, 227, 96, 3, 11, 11, 0, 4, pre_path + "/data1.bin", pre_path + "/filter1.bin");
    test_conv(1, 96, 27, 27, 256, 96, 5, 5, 2, 1, pre_path + "/data2.bin", pre_path + "/filter2.bin");
    test_conv(1, 256, 13, 13, 384, 256, 3, 3, 1, 1, pre_path + "/data3.bin", pre_path + "/filter3.bin");
    test_conv(1, 384, 13, 13, 384, 384, 3, 3, 1, 1, pre_path + "/data4.bin", pre_path + "/filter4.bin");
    test_conv(1, 384, 13, 13, 256, 384, 3, 3, 1, 1, pre_path + "/data5.bin", pre_path + "/filter5.bin");
}


void test_vgg16_matmul() {
    std::string pre_path = "../data/vgg";
    test_matmul_conv(1, 3, 224, 224, 64, 3, 3, 3, 1, 1, pre_path + "/a1.bin", pre_path + "/b1.bin");
    test_matmul_conv(1, 64, 224, 224, 64, 64, 3, 3, 1, 1, pre_path + "/a2.bin", pre_path + "/b2.bin");
    test_matmul_conv(1, 64, 112, 112, 128, 64, 3, 3, 1, 1, pre_path + "/a3.bin", pre_path + "/b3.bin");
    test_matmul_conv(1, 128, 112, 112, 128, 128, 3, 3, 1, 1, pre_path + "/a4.bin", pre_path + "/b4.bin");
    test_matmul_conv(1, 128, 56, 56, 256, 128, 3, 3, 1, 1, pre_path + "/a5.bin", pre_path + "/b5.bin");
    test_matmul_conv(1, 256, 56, 56, 256, 256, 3, 3, 1, 1, pre_path + "/a6.bin", pre_path + "/b6.bin");
    test_matmul_conv(1, 256, 56, 56, 256, 256, 3, 3, 1, 1, pre_path + "/a7.bin", pre_path + "/b7.bin");
    test_matmul_conv(1, 256, 28, 28, 512, 256, 3, 3, 1, 1, pre_path + "/a8.bin", pre_path + "/b8.bin");
    test_matmul_conv(1, 512, 28, 28, 512, 512, 3, 3, 1, 1, pre_path + "/a9.bin", pre_path + "/b9.bin");
    test_matmul_conv(1, 512, 28, 28, 512, 512, 3, 3, 1, 1, pre_path + "/a10.bin", pre_path + "/b10.bin");
    test_matmul_conv(1, 512, 14, 14, 512, 512, 3, 3, 1, 1, pre_path + "/a11.bin", pre_path + "/b11.bin");
    test_matmul_conv(1, 512, 14, 14, 512, 512, 3, 3, 1, 1, pre_path + "/a12.bin", pre_path + "/b12.bin");
    test_matmul_conv(1, 512, 14, 14, 512, 512, 3, 3, 1, 1, pre_path + "/a13.bin", pre_path + "/b13.bin");
}

void test_resnet18_matmul() {
    std::string pre_path = "../data/resnet";
    test_matmul_conv(1, 3, 224, 224, 64, 3, 7, 7, 3, 2, pre_path + "/a1.bin", pre_path + "/b1.bin");
    test_matmul_conv(1, 64, 56, 56, 64, 64, 3, 3, 1, 1, pre_path + "/a2.bin", pre_path + "/b2.bin");
    test_matmul_conv(1, 64, 56, 56, 64, 64, 3, 3, 1, 1, pre_path + "/a3.bin", pre_path + "/b3.bin");
    test_matmul_conv(1, 64, 56, 56, 64, 64, 3, 3, 1, 1, pre_path + "/a4.bin", pre_path + "/b4.bin");
    test_matmul_conv(1, 64, 56, 56, 64, 64, 3, 3, 1, 1, pre_path + "/a5.bin", pre_path + "/b5.bin");
    test_matmul_conv(1, 64, 56, 56, 128, 64, 3, 3, 1, 2, pre_path + "/a6.bin", pre_path + "/b6.bin");
    test_matmul_conv(1, 128, 28, 28, 128, 128, 3, 3, 1, 1, pre_path + "/a7.bin", pre_path + "/b7.bin");
    test_matmul_conv(1, 128, 28, 28, 128, 128, 3, 3, 1, 1, pre_path + "/a8.bin", pre_path + "/b8.bin");
    test_matmul_conv(1, 128, 28, 28, 128, 128, 3, 3, 1, 1, pre_path + "/a9.bin", pre_path + "/b9.bin");
    test_matmul_conv(1, 128, 28, 28, 256, 128, 3, 3, 1, 2, pre_path + "/a10.bin", pre_path + "/b10.bin");
    test_matmul_conv(1, 256, 14, 14, 256, 256, 3, 3, 1, 1, pre_path + "/a11.bin", pre_path + "/b11.bin");
    test_matmul_conv(1, 256, 14, 14, 256, 256, 3, 3, 1, 1, pre_path + "/a12.bin", pre_path + "/b12.bin");
    test_matmul_conv(1, 256, 14, 14, 256, 256, 3, 3, 1, 1, pre_path + "/a13.bin", pre_path + "/b13.bin");
    test_matmul_conv(1, 256, 14, 14, 512, 256, 3, 3, 1, 2, pre_path + "/a14.bin", pre_path + "/b14.bin");
    test_matmul_conv(1, 512, 7, 7, 512, 512, 3, 3, 1, 1, pre_path + "/a15.bin", pre_path + "/b15.bin");
    test_matmul_conv(1, 512, 7, 7, 512, 512, 3, 3, 1, 1, pre_path + "/a16.bin", pre_path + "/b16.bin");
    test_matmul_conv(1, 512, 7, 7, 512, 512, 3, 3, 1, 1, pre_path + "/a17.bin", pre_path + "/b17.bin");
}

void test_alexnet_matmul() {
    std::string pre_path = "../data/alex";
    test_matmul_conv(1, 3, 227, 227, 96, 3, 11, 11, 0, 4, pre_path + "/a1.bin", pre_path + "/b1.bin");
    test_matmul_conv(1, 96, 27, 27, 256, 96, 5, 5, 2, 1, pre_path + "/a2.bin", pre_path + "/b2.bin");
    test_matmul_conv(1, 256, 13, 13, 384, 256, 3, 3, 1, 1, pre_path + "/a3.bin", pre_path + "/b3.bin");
    test_matmul_conv(1, 384, 13, 13, 384, 384, 3, 3, 1, 1, pre_path + "/a4.bin", pre_path + "/b4.bin");
    test_matmul_conv(1, 384, 13, 13, 256, 384, 3, 3, 1, 1, pre_path + "/a5.bin", pre_path + "/b5.bin");
}

int test_bg(int d_n, int d_c, int d_h, int d_w, int f_n, int f_c, int f_h, int f_w, int padding, int stride) {
    auto data = new Tensor4d(d_n, d_c, d_h, d_w);
    data->generateRandData(5);
    auto filter = new Tensor4d(f_n, f_c, f_h, f_w);
    filter->generateRandData(5);
    auto param = new ConvParam(data, filter, padding, stride, 1);
    auto ret = sparse_conv(param);
    printf("n:%d, c:%d, h: %d, w:%d\n", ret->getN(), ret->getC(), ret->getH(), ret->getW());
    return 0;
}

void test_sparse() {
    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_int_distribution<unsigned> u(1, 50); // 闭区间
    int m = 64, n = 16, k = 27;
    float *hA = new float[m * k], *hB = new float[k * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j+=2) {
            int rand = u(e);
            if (j + 1 >= k) {
                hA[i * k + j] = 0;
                continue;
            }
            hA[i * k + j + (int(rand) % 2)] = 0;
            hA[i * k + j + 1 - (int(rand) % 2)] = int(rand);
        }
    }
    for (int i = 0; i < k * n; i++) hB[i] = u(e);

//    printf("A: \n");
//    for (int i = 0; i < m; i++) {
//        for (int j = 0; j < k; j++) {
//            printf("%f ", __half2float(hA[i * k + j]));
//        }
//        printf("\n");
//    }

    half *dA, *dB, *dC, *dD;
    cudaMalloc((void **)&dA, m * k * sizeof(half));
    cudaMalloc((void **)&dB, k * n * sizeof(half));
    cudaMalloc((void **)&dC, m * n * sizeof(half));
    cudaMalloc((void **)&dD, m * n * sizeof(half));
    cudaMemcpy(dA, hA, m * k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, m * n * sizeof(half));
    cudaMemset(dD, 0, m * n * sizeof(half));

    MatrixParam *p = new MatrixParam(m, k, n);

    spmma_matmul(dA, dB, m, k, n, false, dD, p);

    p->checkCorrect(false);
}


int main() {
    test_resnet18();

    // todo: time
}