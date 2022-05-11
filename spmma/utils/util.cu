#include"util.cuh"

short convertIdx2Binary(int *index, int len) {
    short ret = 0;
    for (int i = len - 1; i >= 0; i--) {
        int item = index[i];
        ret = ret << 2;
        ret |= item;
    }
    return ret;
}

size_t get_cmpr_size(int row, int col) {
    int row_blocks = row % 32 ? row / 32 + 1 : row / 32;
    int col_blocks = col % 32 ? col / 32 + 1 : col / 32;
    return row_blocks * col_blocks * 32 * 32 / 8 > 256 ? row_blocks * col_blocks * 32 * 32 / 8 : 256;
}

bool checkGPU() {
    int major_cc, minor_cc;
    cudaDeviceGetAttribute(&major_cc, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor_cc, cudaDevAttrComputeCapabilityMinor, 0);
    if (!(major_cc == 8 && minor_cc == 0) && !(major_cc == 8 && minor_cc == 6)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6 current: %d.%d\n\n",
                    major_cc, minor_cc);
        return false;
    }
    return true;
}

//template <typename Dtype>
//void padMatrix(const Dtype* src, int row, int col, Dtype *dest, int row_padding, int col_padding) {
//    CUDA_CHECK( cudaMemset(dest, 0, row_padding * col_padding * sizeof(Dtype)) )
//    if (col == col_padding) {
//        CUDA_CHECK( cudaMemcpy(dest, src, row * col_padding * sizeof(Dtype), cudaMemcpyDeviceToDevice) )
//    } else {
//        // spitch指定想要复制的矩阵的本身的宽度 width指定需要复制的宽度 dpitch指定赋值到dest的宽度
//        CUDA_CHECK( cudaMemcpy2D(dest, col_padding * sizeof(Dtype), src, col * sizeof(Dtype), col * sizeof(Dtype), row, cudaMemcpyDeviceToDevice) )
//    }
//}
//
//template <typename Dtype>
//void restoreMatrix(const Dtype* src, int row, int col, Dtype *dest, int row_restore, int col_restore, bool toDevice) {
//    auto direction = toDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
//    if (col == col_restore) {
//        CUDA_CHECK( cudaMemcpy(dest, src, row_restore * col * sizeof(Dtype), direction) )
//    } else {
//        CUDA_CHECK(cudaMemcpy2D(dest, col_restore * sizeof(Dtype), src, col * sizeof(Dtype), col_restore * sizeof(Dtype), row_restore, direction) )
//    }
//}


void float2half_array(float *in, half *out, int totalSize) {
    for (int i = 0; i < totalSize; i++) out[i] = __float2half(in[i]);
}

void half2float_array(half *in, float *out, int totalSize) {
    for (int i = 0; i < totalSize; i++) out[i] = __half2float(in[i]);
}