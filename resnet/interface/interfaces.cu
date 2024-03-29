//
// Created by dbettkk on 2022/8/1.
//

#include "interfaces.cuh"

__global__ void ReLU(half *in, half *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    half zero = 0;
    if (idx < size) {
        out[idx] = in[idx] > zero ? in[idx] : zero;
    }
}

void conv2d_device_spmma(half *feature, half *kernel, int batch, int in_c, int out_c,
                   int f_w, int f_h, int k_w, int k_h, int stride, int padding, half *out) {
    // 0. malloc
    int out_w = (f_w + 2 * padding - k_w) / stride + 1;
    int out_h = (f_h + 2 * padding - k_h) / stride + 1;
    half *im2col_out, *gemm_out, *col_im_out;
    CHECK_CUDA(cudaMalloc(&im2col_out, sizeof(half) * batch * out_w * out_h * in_c * k_w * k_h))
    CHECK_CUDA(cudaMalloc(&gemm_out, sizeof(half) * batch * out_w * out_h * out_c))
    CHECK_CUDA(cudaMalloc(&col_im_out,  sizeof(half) * out_c * batch * out_w * out_h))
    // 1. im2col
    im2col_gpu(feature, batch, in_c, f_h, f_w, k_h, k_w, padding, padding,
                     stride, stride, 1, 1, im2col_out);
    // im2col_cudnn(feature, batch, in_c, out_c, f_h, f_w, k_h, k_w, stride, padding, im2col_out);
    // 2. gemm
    sparse_mma_gemm_device(kernel, im2col_out, out_c, in_c * k_h * k_w, batch * out_w * out_h, false, gemm_out);

    // 3. col2im
    int num_kernels = batch * out_w * out_h;
    im2col_rev_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, gemm_out, batch, out_c, out_h, out_w, col_im_out);

    // 4. copy to output
    CHECK_CUDA(cudaMemcpy(out, col_im_out, sizeof(half) * out_c * batch * out_w * out_h, cudaMemcpyDeviceToDevice))

    // 5. free
    CHECK_CUDA(cudaFree(im2col_out))
    CHECK_CUDA(cudaFree(gemm_out))
    CHECK_CUDA(cudaFree(col_im_out))
}

void conv2d_device_cudnn(half *feature, half *kernel, int batch, int in_c, int out_channel,
                         int f_w, int f_h, int k_w, int k_h, int stride, int padding, half *out) {
    // read from bin_file
    int data_n = batch, data_c = in_c, data_h = f_h, data_w = f_w;
    int kernel_n = out_channel, kernel_c = in_c, kernel_h = k_h, kernel_w = k_w;

    size_t data_size = data_n * data_c * data_h * data_w * sizeof(half);
    size_t kernel_size = kernel_n * kernel_c * kernel_w * kernel_h * sizeof(half);

    int dilation = 1;

    //handle
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle))

    // input
    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           data_n, data_c, data_h, data_w)) // n, c, h, w


    // kernel
    //printTensor(kernel, kernel_n, kernel_c, kernel_w, kernel_h, "kernel: ");
    cudnnFilterDescriptor_t kernel_descriptor;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor))
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
                                           kernel_n, kernel_c, kernel_h, kernel_w))


    // convolution descriptor
    cudnnConvolutionDescriptor_t conv_descriptor;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor))
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_descriptor,
                                                padding, padding, // zero-padding
                                                stride, stride, // stride
                                                dilation, dilation, // dilation 卷积核膨胀 膨胀后用0填充空位
            // 卷积是需要将卷积核旋转180°再进行后续的 -> CUDNN_CONVOLUTION
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT))


    // output
    int out_n, out_c, out_h, out_w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_descriptor, input_descriptor, kernel_descriptor,
                                                      &out_n, &out_c, &out_h, &out_w))
    //printf("output: %d * %d * %d * %d\n", out_n, out_c, out_h, out_w);
    //size_t out_size = out_n * out_c * out_h * out_w * sizeof(half);

    cudnnTensorDescriptor_t output_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           out_n, out_c, out_h, out_w))


    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT; // no support
    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;


    // workspace size && allocate memory
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                        input_descriptor,
                                                        kernel_descriptor,
                                                        conv_descriptor,
                                                        output_descriptor,
                                                        algo,
                                                        &workspace_size))

    void *workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size))

    // convolution
    auto alpha = 1.0f, beta = 0.0f;

    // calculate
    CHECK_CUDNN(cudnnConvolutionForward(handle,
                                        &alpha, input_descriptor, feature,
                                        kernel_descriptor, kernel,
                                        conv_descriptor, algo,
                                        workspace, workspace_size,
                                        &beta, output_descriptor, out))

    // destroy
    CHECK_CUDA(cudaFree(workspace))

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor))
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_descriptor))
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_descriptor))
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor))

    CHECK_CUDNN(cudnnDestroy(handle))

    //printTensor(output, out_n, out_c, out_h, out_w, "output: ");
    // 数据量大时 测试正确性使用
//    for (int i = 0; i < out_h; i++) {
//        for (int j = 0; j < out_w; j++) {
//            printf("%d", __half2int_rz(output[i * out_w + j]));
//        }
//        printf("\n");
//    }
}

void conv2d_device_cusparse(half *feature, half *kernel, int batch, int in_c, int out_c,
                         int f_w, int f_h, int k_w, int k_h, int stride, int padding, half *out) {
    // 0. malloc
    int out_w = (f_w + 2 * padding - k_w) / stride + 1;
    int out_h = (f_h + 2 * padding - k_h) / stride + 1;
    half *im2col_out, *gemm_out, *col_im_out;
    CHECK_CUDA(cudaMalloc(&im2col_out, sizeof(half) * batch * out_w * out_h * in_c * k_w * k_h))
    CHECK_CUDA(cudaMalloc(&gemm_out, sizeof(half) * batch * out_w * out_h * out_c))
    CHECK_CUDA(cudaMalloc(&col_im_out,  sizeof(half) * out_c * batch * out_w * out_h))
    // 1. im2col
    im2col_gpu(feature, batch, in_c, f_h, f_w, k_h, k_w, padding, padding,
               stride, stride, 1, 1, im2col_out);
    // im2col_cudnn(feature, batch, in_c, out_c, f_h, f_w, k_h, k_w, stride, padding, im2col_out);
    // 2. gemm
    cusparse_gemm_csr_device(kernel, im2col_out, out_c, in_c * k_h * k_w, batch * out_w * out_h, gemm_out);
    //cusparse_gemm_coo_device(kernel, im2col_out, out_c, in_c * k_h * k_w, batch * out_w * out_h, gemm_out);

    // 3. col2im
    int num_kernels = batch * out_w * out_h;
    im2col_rev_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, gemm_out, batch, out_c, out_h, out_w, col_im_out);

    // 4. copy to output
    CHECK_CUDA(cudaMemcpy(out, col_im_out, sizeof(half) * out_c * batch * out_w * out_h, cudaMemcpyDeviceToDevice))

    // 5. free
    CHECK_CUDA(cudaFree(im2col_out))
    CHECK_CUDA(cudaFree(gemm_out))
    CHECK_CUDA(cudaFree(col_im_out))
}

void bn_cudnn(half *feature, int batch, int channel, int width, int height, half *out) {
    // handle
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle))
    // bn mode
    cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
    float alpha = 1.0, beta = 0.0;
    double epsilon = 0.00001;
    // input
    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           batch, channel, height, width)) // n, c, h,

    // output
    cudnnTensorDescriptor_t output_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           batch, channel, height, width))

    // get bnScaleBiasMeanVarDesc
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc))
    CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc, input_descriptor, mode))

    // param
    auto *scale = new float[channel];
    auto *bias = new float[channel];
    for (int i = 0; i < channel; i++) {
        scale[i] = 1;
        bias[i] = 0;
    }
    float *d_scale, *d_bias;
    cudaMalloc(&d_scale, sizeof(float) * channel);
    cudaMalloc(&d_bias, sizeof(float) * channel);
    cudaMemcpy(d_scale, scale, sizeof(float) * channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, sizeof(float) * channel, cudaMemcpyHostToDevice);

    // 计算
    CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(handle, mode, &alpha, &beta, input_descriptor, feature, output_descriptor,
                                           out, bnScaleBiasMeanVarDesc, d_scale, d_bias, 0.0,
                                           nullptr, nullptr, epsilon, nullptr, nullptr))
    // free
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor))
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_descriptor))
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc))

    CHECK_CUDNN(cudnnDestroy(handle))
}

void pool_cudnn(half *feature, int batch, int channel, int width, int height, half *out, int window_size,
              int padding, int stride, int modes) {
    /* mode: 1 -> maxpool | 其他 -> avgpool */
    // handle
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle))
    float alpha = 1.0, beta = 0.0;
    // mode
    cudnnPoolingMode_t mode = modes == 1 ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    cudnnNanPropagation_t np = CUDNN_NOT_PROPAGATE_NAN;
    // pool
    cudnnPoolingDescriptor_t pool;
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pool))
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(pool, mode, np, window_size, window_size, padding, padding, stride, stride))

    // input
    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           batch, channel, height, width)) // n, c, h,

    // get output dim
    int out_n, out_c, out_h, out_w;
    CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim(pool, input_descriptor, &out_n, &out_c, &out_h, &out_w))
    // printf("NCHW: %d %d %d %d\n", out_n, out_c, out_h, out_w);

    // output
    cudnnTensorDescriptor_t output_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           out_n, out_c, out_h, out_w))

    // pool
    CHECK_CUDNN(cudnnPoolingForward(handle, pool, &alpha, input_descriptor, feature, &beta, output_descriptor, out))

    // free
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor))
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_descriptor))
    CHECK_CUDNN(cudnnDestroyPoolingDescriptor(pool))

    CHECK_CUDNN(cudnnDestroy(handle))

}

void softmax_cudnn(half *feature, int batch, int channel, int width, int height, half *out) {
    // handle
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle))

    float alpha = 1.0, beta = 0.0;

    // input
    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           batch, channel, height, width)) // n, c, h,
    // output
    cudnnTensorDescriptor_t output_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           batch, channel, height, width))

    // softmax
    CHECK_CUDNN(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
                                    input_descriptor, feature, &beta, output_descriptor, out))

    // free
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor))
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_descriptor))

    CHECK_CUDNN(cudnnDestroy(handle))
}

void im2col_cudnn(half *feature, int batch, int in_c, int out_c, int f_h, int f_w, int k_h, int k_w, int stride, int padding, half *out) {
    // handle
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle))
    // input
    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           batch, in_c, f_h, f_w)) // n, c, h, w


    // kernel
    //printTensor(kernel, kernel_n, kernel_c, kernel_w, kernel_h, "kernel: ");
    cudnnFilterDescriptor_t kernel_descriptor;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor))
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
                                           out_c, in_c, k_h, k_w))

    // convolution descriptor
    cudnnConvolutionDescriptor_t conv_descriptor;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor))
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_descriptor,
                                                padding, padding, // zero-padding
                                                stride, stride, // stride
                                                1, 1, // dilation 卷积核膨胀 膨胀后用0填充空位
            // 卷积是需要将卷积核旋转180°再进行后续的 -> CUDNN_CONVOLUTION
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT))

    // im2col
    CHECK_CUDNN(cudnnIm2Col(handle, input_descriptor, feature, kernel_descriptor, conv_descriptor, out))

    // free
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor))
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor))
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_descriptor))
    CHECK_CUDNN(cudnnDestroy(handle))
}