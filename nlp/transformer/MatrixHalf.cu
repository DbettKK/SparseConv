//
// Created by dbettkk on 2022/7/23.
//

#include "MatrixHalf.cuh"

MatrixHalf::MatrixHalf(half *matrix, int batch, int row, int col) : matrix(matrix), batch(batch), row(row), col(col) {}

MatrixHalf::MatrixHalf(int batch, int row, int col, bool is_device) : batch(batch), row(row), col(col) {
    if (!is_device) this->matrix = new half[batch * row * col];
    else CHECK_CUDA(cudaMalloc(&this->matrix, sizeof(half) * batch * row * col))
}

MatrixHalf::MatrixHalf(int batch, int row, int col, bool is_device, half init) : batch(batch), row(row), col(col) {
    half *tmp = new half[batch * row * col];
    for (int i = 0; i < batch * row * col; i++) tmp[i] = init;
    if (is_device) {
        CHECK_CUDA(cudaMalloc(&matrix, sizeof(half) * row * col * batch));
        CHECK_CUDA(cudaMemcpy(matrix, tmp, sizeof(half) * row * col * batch, cudaMemcpyHostToDevice));
        delete[] tmp;
    } else {
        matrix = tmp;
    }
}

half *MatrixHalf::getMatrix() const {
    return matrix;
}

void MatrixHalf::setMatrix(half *matrix) {
    MatrixHalf::matrix = matrix;
}

int MatrixHalf::getBatch() const {
    return batch;
}

void MatrixHalf::setBatch(int batch) {
    MatrixHalf::batch = batch;
}

int MatrixHalf::getRow() const {
    return row;
}

void MatrixHalf::setRow(int row) {
    MatrixHalf::row = row;
}

int MatrixHalf::getCol() const {
    return col;
}

void MatrixHalf::setCol(int col) {
    MatrixHalf::col = col;
}

void MatrixHalf::gemm(MatrixHalf *item, MatrixHalf *out) {
    if (this->batch != out->getBatch()) {
        printf("source batch_size should be equaled to the target batch_size!");
        return;
    }
    cublas_gemm_device(this->matrix, item->matrix, this->row, this->col, item->col, out->matrix);
    //sparse_mma_gemm_device(this->matrix, item->matrix, this->row, this->col, item->col, true, out->matrix);
    //dim3 grid(16, 16);
    //dim3 block(32, 32);
    //gemm_simple<<<grid, block>>>(this->matrix, item->matrix, row, col, item->col, out->matrix);
}

void MatrixHalf::gemm_batches(MatrixHalf *item, MatrixHalf *out, bool is_single_batch) {
    // is_single_batch: item的batch数是否为1
    if (is_single_batch) {
        for (int i = 0; i < batch; i++) {
            cublas_gemm_device(matrix + i * row * col, item->matrix, row, col, item->col,
                               out->matrix + i * out->row * out->col);
        }
    } else {
        for (int i = 0; i < batch; i++) {
            cublas_gemm_device(matrix + i * row * col, item->matrix + i * item->row * item->col,
                               row, col, item->col, out->matrix + i * out->row * out->col);
        }
    }
}

int MatrixHalf::getSize() const {
    return this->batch * this->col * this->row;
}

void MatrixHalf::reshape(MatrixHalf *out, int heads) const {
    dim3 block(this->batch, heads);
    dim3 thread(this->row, this->col / heads);
    reshape_multi_head<<<block, thread>>>(this->matrix, out->matrix, this->row, this->col, heads);
}

void MatrixHalf::transpose(MatrixHalf *out) {
    dim3 thread(this->row, this->col);
    transpose_half<<<1, thread>>>(this->matrix, out->matrix, this->row, this->col);
}

void MatrixHalf::softmax() {
    softmax_half<<<this->col, this->row>>>(this->matrix, this->row, this->col);
}

void MatrixHalf::print(const std::string& msg, bool is_device) {
    std::cout << msg << std::endl;
    if (is_device) {
        half *tmp = new half[row * col];
        CHECK_CUDA(cudaMemcpy(tmp, matrix, sizeof(half) * row * col, cudaMemcpyDeviceToHost));
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                printf("%.2f ", __half2float(tmp[i * col + j]));
            }
            printf("\n");
        }
        delete[] tmp;
    } else {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                printf("%.2f ", __half2float(matrix[i * col + j]));
            }
            printf("\n");
        }
    }

}

void MatrixHalf::free_matrix() {
    cudaError_t st = cudaFree(matrix);
    if (st != cudaSuccess) {
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",
               __FILE__, __LINE__, cudaGetErrorString(st), st);
        return;
    }
    //CHECK_CUDA(cudaFree(matrix))
}

 void MatrixHalf::print_device(half *item, int row, int col) {
    // std::cout << msg << std::endl;
    half *tmp = new half[row * col];
    CHECK_CUDA(cudaMemcpy(tmp, item, sizeof(half) * row * col, cudaMemcpyDeviceToHost));
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%.2f ", __half2float(tmp[i * col + j]));
        }
        printf("\n");
    }
    delete[] tmp;
}

void MatrixHalf::relu() {
    dim3 grid(row / 32 + 1, col / 32 + 1);
    dim3 block(32, 32);
    relu_half<<<grid, block>>>(matrix, row, col);
}

void MatrixHalf::addMatrix(MatrixHalf *add, MatrixHalf *out) {
    if (this->row != add->row || this->col != add->col) {
        printf("Error: matrix add need to be the same shape!\n");
        return;
    }
    matrix_add<<<batch, row * col>>>(matrix, add->matrix, out->matrix, batch * row * col);
}

void MatrixHalf::copyTo(MatrixHalf *out) {
    cudaMemcpy(out->getMatrix(), matrix, sizeof(half) * out->getSize(), cudaMemcpyDeviceToDevice);
}



