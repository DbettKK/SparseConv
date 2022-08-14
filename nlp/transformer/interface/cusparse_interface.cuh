//
// Created by dbettkk on 2022/8/14.
//

#ifndef SPARSECONV_CUSPARSE_INTERFACE_CUH
#define SPARSECONV_CUSPARSE_INTERFACE_CUH

#include "kernels_transformer.cuh"

void cusparse_gemm_csr_device(half *sp_A, half *d_B, int m, int k, int n, half *output);

void cusparse_gemm_blocked_device_test();

void cusparse_gemm_csr_device_test();

#endif //SPARSECONV_CUSPARSE_INTERFACE_CUH
