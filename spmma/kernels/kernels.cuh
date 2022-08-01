//
// Created by dbettkk on 2022/3/31.
//

#ifndef SPARSECONVOLUTION_KERNELS_CUH
#define SPARSECONVOLUTION_KERNELS_CUH

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",          \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        return;                                                                \
    }                                                                          \
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);  i += blockDim.x * gridDim.x)

#define CUDA_POST_KERNEL_CHECK CHECK_CUDA(cudaPeekAtLastError())

// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im, const int data_n, const int channel,
                                  const int height, const int width, const int kernel_h, const int kernel_w,
                                  const int pad_h, const int pad_w,
                                  const int stride_h, const int stride_w,
                                  const int dilation_h, const int dilation_w,
                                  const int height_col, const int width_col, Dtype* data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        for (int idn = 0; idn < data_n; idn++) {
            const int h_index = index / width_col;
            const int h_col = h_index % height_col;
            const int w_col = index % width_col;
            const int c_im = h_index / height_col;
            const int c_col = c_im * kernel_h * kernel_w;
            const int h_offset = h_col * stride_h - pad_h;
            const int w_offset = w_col * stride_w - pad_w;
            Dtype* data_col_ptr = data_col;
            data_col_ptr += idn * height_col * width_col + (c_col * height_col * data_n + h_col) * width_col  + w_col;   // 确定输出的pointer的位置
            const Dtype* data_im_ptr = data_im;
            data_im_ptr += idn * channel * height * width + (c_im * height + h_offset) * width + w_offset;   // 确定图像的位置

            for (int i = 0; i < kernel_h; ++i) {
                for (int j = 0; j < kernel_w; ++j) {
                    int h_im = h_offset + i * dilation_h;
                    int w_im = w_offset + j * dilation_w;
                    *data_col_ptr =
                            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                            data_im_ptr[i * dilation_h * width + j * dilation_w] : __int2half_rn(0);
                    data_col_ptr += data_n * height_col * width_col;
                }
            }
        }
    }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int data_n, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w, Dtype* data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h -
                      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w -
                     (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    // NOLINT_NEXT_LINE(whitespace/operators)
    im2col_gpu_kernel<Dtype> <<< GET_BLOCKS(num_kernels), CUDA_NUM_THREADS >>>(
            num_kernels, data_im, data_n, channels, height, width, kernel_h, kernel_w, pad_h,
            pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, data_col);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void im2col_rev_kernel(
        const int n, const Dtype *data, int data_n, int kernel_n, int out_h, int out_w, Dtype *out) {
    // row: kernel_n
    // col: n * out_h * out_w
    // n * out_h * out_w个线程
    CUDA_KERNEL_LOOP(index, n) {
        // 每个thread负责一个卷积核对应位置的所有channel
        int line = index % (data_n * out_h * out_w);
        int n_index = index / (out_h * out_w);
        int h_index = (index - n_index * (out_h * out_w)) / out_w;
        int w_index = (index - n_index * (out_h * out_w)) - h_index * out_w;
        for (int i = 0; i < kernel_n; i++) {
            out[n_index * kernel_n * out_h * out_w + i * out_h * out_w + h_index * out_w + w_index] = data[i * data_n * out_h * out_w + line];
            //half a = data[i * data_n * out_h * out_w + line];
            //out[n_index * kernel_n * out_h * out_w + i * out_h * out_w + h_index * out_w + w_index] = 0;
        }
    }
}
#endif //SPARSECONVOLUTION_KERNELS_CUH
