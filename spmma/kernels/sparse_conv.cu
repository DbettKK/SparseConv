//
// Created by dbettkk on 2022/3/30.
//
#include"sparse_conv.cuh"

Tensor4d *sparse_conv(ConvParam *param) {
    // time
    auto time1 = new CudaTime();
    auto time2 = new CudaTime();
    auto time3 = new CudaTime();

    time1->initAndStart();
    time3->initAndStart();

    float *d_kernel, *d_im2col;
    CUDA_CHECK( cudaMalloc((void **)&d_kernel, param->getKernel()->getTotalSize() * sizeof(float)) )
    CUDA_CHECK( cudaMalloc((void **)&d_im2col, param->getIm2colSize() * sizeof(float)) )
    param->im2colGPU(d_kernel, d_im2col);

    float im2colTime = time1->endAndGetTime();

    float *dD;
    CUDA_CHECK( cudaMalloc((void **)&dD, sizeof(float) * param->getM() * param->getN()) )
    auto mm_out = new MatrixParam(param->getN(), param->getK(), param->getM());

    spmma_matmul(d_kernel, d_im2col, param->getN(), param->getK(), param->getM(), false,
                 dD, mm_out);


    CUDA_CHECK( cudaFree(d_kernel) )
    CUDA_CHECK( cudaFree(d_im2col) )

    time2->initAndStart();

    Tensor4d* ret = param->col2imGPU(dD);

    float col2imTime = time2->endAndGetTime();
    float totalTime = time3->endAndGetTime();

    CUDA_CHECK( cudaFree(dD) )

    // time
    std::ofstream out("../data/spmma_time.txt", std::ios::app);
    out << "im2col: " << im2colTime << "ms, col2im: " << col2imTime << "ms\n";
    out << "total: " << totalTime << "ms\n";
    out.close();


    // check correct
    MatrixParam* check_im2col = param->im2col();
    auto realD = transpose<float>(mm_out->getMatD(), param->getN(), param->getM());
    check_im2col->setMatD(realD);
    check_im2col->checkCorrect(false);
    return ret;
//    // 数据量大时 测试正确性用
////    for (int i = 0; i < 1; i++) {
////        for (int j = 0; j < 1; j++) {
////            for (int ki = 0; ki < ret->getH(); ki++) {
////                for (int v = 0; v < ret->getW(); v++) {
////                    printf("%d ", __half2int_rz(ret->getTensor()[ki * ret->getW() + v]));
////                }
////                printf("\n");
////            }
////        }
////    }
//
//    CUDA_CHECK( cudaFree(d_kernel) )
//    CUDA_CHECK( cudaFree(d_im2col) )
//    CUDA_CHECK( cudaFree(dD) )
//
//    delete mm_out;
//    delete check_im2col;
//
//    return ret;
}
