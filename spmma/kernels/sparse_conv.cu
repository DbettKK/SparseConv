//
// Created by dbettkk on 2022/3/30.
//
#include"sparse_conv.cuh"

Tensor4d *sparse_conv(ConvParam *param) {
    half *d_kernel, *d_im2col;
    CUDA_CHECK( cudaMalloc((void **)&d_kernel, param->getKernel()->getTotalSize() * sizeof(half)) )
    CUDA_CHECK( cudaMalloc((void **)&d_im2col, param->getIm2colSize() * sizeof(half)) )

    param->im2colGPU(d_kernel, d_im2col);

    half *dD;
    CUDA_CHECK( cudaMalloc((void **)&dD, sizeof(half) * param->getM() * param->getN()) )

    //auto *mm_out = new MatrixParam(param->getN(), param->getK(), param->getM());

    spmma_matmul(d_kernel, d_im2col, param->getN(), param->getK(), param->getM(), false, dD, nullptr);

    CUDA_CHECK( cudaFree(d_kernel) )
    CUDA_CHECK( cudaFree(d_im2col) )

    Tensor4d* ret = param->col2imGPU(dD);

    CUDA_CHECK( cudaFree(dD) )

    return ret;
    // check correct
    // 如果存在需要padding 这个是无法检验正确性的！！！
//    MatrixParam* check_im2col = param->im2col();
//
//    half *realD = transpose<half>(mm_out->getMatD(), param->getN(), param->getM());
//    check_im2col->setMatD(realD);
//    check_im2col->checkCorrect(false);
    //return ret;
//    // 数据量大时 测试正确性用
//    for (int i = 0; i < 1; i++) {
//        for (int j = 0; j < 1; j++) {
//            for (int ki = 0; ki < ret->getH(); ki++) {
//                for (int v = 0; v < ret->getW(); v++) {
//                    printf("%d ", __half2int_rz(ret->getTensor()[ki * ret->getW() + v]));
//                }
//                printf("\n");
//            }
//        }
//    }
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
