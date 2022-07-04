//
// Created by dbettkk on 2022/6/10.
//
#include "wmma.sp.cuh"

static const int M = 16;
static const int N = 8;
static const int K = 16;

__global__ void sparse_mma(half *d, half *a, half *b, half *c, const uint32_t *metadata_p) {
    uint32_t tid = threadIdx.x;
    uint32_t metadata = metadata_p[tid / 4];
    __half *a_ptr = a + (tid % 4) * 2 + (tid / 4) * 8;
    __half *b_ptr = b + (tid % 4) * 2 * 8 + tid / 4;
    __half *c_ptr = c + (tid % 4) * 2 + (tid / 4) * 8;
    __half *d_ptr = d + (tid % 4) * 2 + (tid / 4) * 8;
    asm volatile("{\n\t"
                 ".reg .f16 %Ra_single<4>, %Rb_single<4>;\n\t"
                 ".reg .f16x2 %Ra<2>, %Rb<2>, %Rc<2>, %Rd<2>;\n\t"
                 "ld.global.ca.b32 %Ra0, [%1];\n\t"
                 "ld.global.ca.b32 %Ra1, [%1 + 128];\n\t"
                 "ld.global.ca.b16 %Rb_single0, [%2];\n\t"
                 "ld.global.ca.b16 %Rb_single1, [%2 + 16];\n\t"
                 "ld.global.ca.b16 %Rb_single2, [%2 + 128];\n\t"
                 "ld.global.ca.b16 %Rb_single3, [%2 + 144];\n\t"
                 "ld.global.ca.b32 %Rc0, [%3];\n\t"
                 "ld.global.ca.b32 %Rc1, [%3 + 128];\n\t"
                 "mov.b32 %Rb0, {%Rb_single0, %Rb_single1};\n\t"
                 "mov.b32 %Rb1, {%Rb_single2, %Rb_single3};\n\t"
                 "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16\n\t"
                 "{%Rd0, %Rd1},\n\t"
                 "{%Ra0, %Ra1},\n\t"
                 "{%Rb0, %Rb1},\n\t"
                 "{%Rc0, %Rc1}, %4, 0x0;\n\t"
                 "st.global.wb.b32 [%0], %Rd0;\n\t"
                 "st.global.wb.b32 [%0 + 128], %Rd1;\n\t"
                 "}\n\t"
            :
            : "l"(d_ptr), "l"(a_ptr), "l"(b_ptr), "l"(c_ptr), "r"(metadata)
            : "memory");
}

/**
 * Compress matA and Get Sparse-Index(meta)
 * index: two rows as a group
 * sparse-index:[0 2 1 2 0 3 1 3, 1 2 1 3 0 3 0 1]
 * bin-meta: reverse(00 10 01 10 00 11 01 11, 01 10 01 11 00 11 00 01) ->
 * (01 00 11 00 11 01 10 01 ,11 01 11 00 10 01 10 00)
 * @param mat 16x16 IN
 * @param mat_cmpr 16x8 OUT
 * @param meta 8 OUT
 */
__device__ void compress_mat(half *mat, half *mat_cmpr, uint32_t *meta) {
    int *bin_meta = new int[M * K / 2];
    int bin_meta_index = 0, mat_cmpr_index = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K / 4; j++) {
            int zero_num = 0;
            int tmp[4] = {1, 2, 3, 4};
            for (int k = 0; k < 4; k++) {
                if (__half2float(mat[i * K + j * 4 + k]) == 0) {
                    tmp[k] *= -1;
                    zero_num++;
                    if (zero_num == 2) break;
                }
            }
            // problem
            for (int k = 0; k < 4; k++) {
                if (tmp[k] > 0) {
                    bin_meta[bin_meta_index++] = k;
                    mat_cmpr[mat_cmpr_index++] = mat[i * K + j * 4 + k];
                    //printf("%d:%d", k, __half2int_rz(mat[i * K + j * 4 + k]));
                }
            }
        }
    }
    for (int i = 0; i < M * K / 2 / 16; i++) {
        int metadata = 0, offset = 0;
        for (int j = 0; j < 16; j++) {
            int index = i * 16 + j;
            metadata |= bin_meta[index] << offset;
            offset += 2;
        }
        meta[i] = metadata;
    }
    delete[] bin_meta;
}

__device__ void split_matB(half *matB, half *out1, half *out2) {
    int out1_index = 0, out2_index = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K / 2; j++) {
            out1[out1_index++] = matB[i * K + j];
        }
        for (int j = K / 2; j < K; j++) {
            out2[out2_index++] = matB[i * K + j];
        }
    }
}

__device__ void merge_out(half *out1, half *out2, half *out) {
    int out1_index = 0, out2_index = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K / 2; j++) {
            out[i * K + j] = out1[out1_index++];
        }
        for (int j = K / 2; j < K; j++) {
            out[i * K + j] = out2[out2_index++];
        }
    }
}

__global__ void compress_mat_multi(half *matA, half *matA_cmpr, uint32_t *metadata, uint32_t *final_meta) {
    // task num: 32
    // m16n16k16
    // metadata[128]
    int tid = threadIdx.x;
    int row_num = tid / 2;
    int col_num = tid % 2;
    int flag[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int init_index = 16 * row_num + 8 * col_num;
    for (int i = 0; i < 8; i++) {
        if (matA[i + init_index] == __float2half(0)) {
            flag[i] = -1;
        }
    }
    int cmpr_index = 8 * row_num + 4 * col_num;
    int meta_index = cmpr_index;
    int cnt = 0;
    for (int i = 0; i < 8; i++) {
        if (flag[i] > 0) {
            matA_cmpr[cmpr_index + cnt] = matA[i + init_index];
            metadata[meta_index + cnt] = i >= 4 ? i - 4 : i;
            cnt++;
        }
    }
    if (tid % 4 == 0) {
        int final_meta_index = tid / 4 * 16;
        int ans = 0, offset = 0;
        for (int i = final_meta_index; i < final_meta_index + 16; i++) {
            ans |= metadata[i] << offset;
            offset += 2;
        }
        final_meta[tid / 4] = ans;
    }
}

/**
 *
 * @param matA 16x16
 * @param matB 16x16
 * @param matC 16x16
 * @param out 16x16
 */
__global__ void wmma_spmma_test(half *matA, half *matB, half *matC1, half *matC2, half *out) {
//    half *matA_cmpr = new half[M * K / 2];
//    half *matB1 = new half[16 * 8];
//    half *matB2 = new half[16 * 8];
//    half *out1 = new half[16 * 8];
//    half *out2 = new half[16 * 8];
//    uint32_t *meta = new uint32_t[8];
//    uint32_t *old_meta = new uint32_t[128];
//    //compress_mat(matA, matA_cmpr, meta);
//    //compress_mat_multi<<<1, 32>>>(matA, matA_cmpr, old_meta, meta);
//    //split_matB(matB, matB1, matB2);
//    //sparse_mma<<<1, 32>>>(out1, matA_cmpr, matB1, matC1, meta);
//    //sparse_mma<<<1, 32>>>(out2, matA_cmpr, matB2, matC2, meta);
//
//    //merge_out(out1, out2, out);
//
//    delete[] matB1;
//    delete[] matB2;
//    delete[] out1;
//    delete[] out2;
}

__global__ void wmma_example(half *a, half *b, half *c) {
    // Declare the fragments m,n,k
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);
    // Perform the matrix multiplication
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    wmma::store_matrix_sync(c, acc_frag, 16, wmma::mem_row_major);
}

void ptx_161616() {
    // random
    std::random_device sd; // sd可以产生一个质量很高的随机数
    std::default_random_engine e(sd());
    std::uniform_int_distribution<unsigned> u(0, 1); // 闭区间
    std::uniform_int_distribution<unsigned> u2(0, 90); // 闭区间

    half *hA = new half[16 * 16];
    half *hB = new half[16 * 16];
    for (int i = 0; i < 16 * 16; i+=4) {
//        int zero_0 = u(e);
//        int zero_1 = u(e);
        hA[i] = 0;
        hA[i + 1] = 1;
        hA[i + 2] = 0;
        hA[i + 3] = 3;
    }
    for (int i = 0; i < 16 * 16; i++) hB[i] = 1;

//    for (int i = 0; i < 16; i++) {
//        for (int j = 0; j < 16; j++) {
//            printf("%d ", __half2int_rz(hA[i * 16 + j]));
//        }
//        printf("\n");
//    }

    half *dA, *dB, *dC1, *dC2, *dOut;

    cudaMalloc(&dA, sizeof(half) * 256);
    cudaMalloc(&dB, sizeof(half) * 256);
    cudaMalloc(&dC1, sizeof(half) * 256);
    cudaMalloc(&dC2, sizeof(half) * 256);
    cudaMalloc(&dOut, sizeof(half) * 256);

    cudaMemset(dC1, 0, sizeof(half) * 256);
    cudaMemset(dC2, 0, sizeof(half) * 256);
    cudaMemset(dOut, 0, sizeof(half) * 256);

    cudaMemcpy(dA, hA, sizeof(half) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(half) * 256, cudaMemcpyHostToDevice);

    for (int i = 0; i < 10; i++) {
        cudaMemset(dC1, 0, sizeof(half) * 256);
        cudaMemset(dC2, 0, sizeof(half) * 256);
        cudaMemset(dOut, 0, sizeof(half) * 256);
        CudaTime time;
        time.initAndStart();
        wmma_spmma_test<<<1, 1>>>(dA, dB, dC1, dC2, dOut);
        float times = time.endAndGetTime();
        printf("%f\n", times);
    }

    half *hOut = new half[16 * 16];
    cudaMemcpy(hOut, dOut, sizeof(half) * 256, cudaMemcpyDeviceToHost);

    // printf("\n");
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%d ", __half2int_rz(hOut[i * 16 + j]));
        }
        printf("\n");
    }
}

void wmma_pure() {
    half *hA = new half[16 * 16];
    half *hB = new half[16 * 16];

    for (int i = 0; i < 16 * 16; i+=4) {
//        int zero_0 = u(e);
//        int zero_1 = u(e);
        hA[i] = 0;
        hA[i + 1] = 1;
        hA[i + 2] = 0;
        hA[i + 3] = 2;
    }
    for (int i = 0; i < 256; i++) hB[i] = 1;

    half *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(half) * 256);
    cudaMalloc(&dB, sizeof(half) * 256);
    cudaMalloc(&dC, sizeof(half) * 256);
    cudaMemcpy(dA, hA, sizeof(half) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(half) * 256, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, sizeof(half) * 256);


    for (int i = 0; i < 10; i++) {
        cudaMemset(dC, 0, sizeof(half) * 256);
        CudaTime time;
        time.initAndStart();
        wmma_example<<<46, 128>>>(dA, dB, dC);
        float times = time.endAndGetTime();
        printf("time: %fms\n", times);
    }


    half *hC = new half[256];
    cudaMemcpy(hC, dC, sizeof(half) * 256, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%d ", __half2int_rz(hC[i * 16 + j]));
        }
        printf("\n");
    }
}

void ptx_pure() {
    half *hA = new half[16 * 8];
    half *hB = new half[16 * 8];
    auto hMeta = new uint32_t[8];
    for (int i = 0; i < 16 * 8; i++) hA[i] = 2;
    for (int i = 0; i < 16 * 8; i++) hB[i] = 1;
    for (int i = 0; i < 8; i++) hMeta[i] = 0b10001000100010001000100010001000;

    half *dA, *dB, *dC, *dOut;
    uint32_t *dMeta;
    cudaMalloc(&dA, sizeof(half) * 128);
    cudaMalloc(&dB, sizeof(half) * 128);
    cudaMalloc(&dC, sizeof(half) * 128);
    cudaMalloc(&dOut, sizeof(half) * 128);
    cudaMalloc(&dMeta, sizeof(uint32_t) * 8);

    cudaMemset(dC, 0, sizeof(half) * 128);
    cudaMemset(dOut, 0, sizeof(half) * 128);

    cudaMemcpy(dA, hA, sizeof(half) * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(half) * 128, cudaMemcpyHostToDevice);
    cudaMemcpy(dMeta, hMeta, sizeof(uint32_t) * 8, cudaMemcpyHostToDevice);

    for (int i = 0; i < 10; i++) {
        cudaMemset(dC, 0, sizeof(half) * 128);
        cudaMemset(dOut, 0, sizeof(half) * 128);
        CudaTime time;
        time.initAndStart();
        sparse_mma<<<46, 128>>>(dOut, dA, dB, dC, dMeta);
        float times = time.endAndGetTime();
        printf("time: %fms\n", times);
    }

    half *hOut = new half[16 * 8];
    cudaMemcpy(hOut, dOut, sizeof(half) * 128, cudaMemcpyDeviceToHost);

    // printf("\n");
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%d ", __half2int_rz(hOut[i * 8 + j]));
        }
        printf("\n");
    }
}

void test_cmpr() {
    auto gen = new DataGenerator();
    half *hA = new half[16 * 16];
    for (int i = 0; i < 16 * 16; i+=4) {
//        int zero_0 = u(e);
//        int zero_1 = u(e);
        hA[i] = 0;
        hA[i + 1] = 1;
        hA[i + 2] = 0;
        hA[i + 3] = 2;
    }

    half *dA, *dA_cmpr;
    uint32_t *metadata, *final_meta;
    cudaMalloc(&dA, sizeof(half) * 256);
    cudaMalloc(&dA_cmpr, sizeof(half) * 128);
    cudaMalloc(&metadata, sizeof(uint32_t) * 16 * 8);
    cudaMalloc(&final_meta, sizeof(uint32_t) * 8);

    cudaMemcpy(dA, hA, sizeof(half) * 256, cudaMemcpyHostToDevice);

    compress_mat_multi<<<1, 32>>>(dA, dA_cmpr, metadata, final_meta);

    half *hA_cmpr = new half[128];
    uint32_t *hMetadata = new uint32_t[128];
    cudaMemcpy(hA_cmpr, dA_cmpr, sizeof(half) * 128, cudaMemcpyDeviceToHost);
    cudaMemcpy(hMetadata, metadata, sizeof(uint32_t) * 128, cudaMemcpyDeviceToHost);

    gen->printMatrix(hA, 16, 16);
    gen->printMatrix(hA_cmpr, 16, 8);

    for (int i = 0; i < 16; i++) printf("%d ", hMetadata[i]);


}

int main() {

    ptx_161616();
}