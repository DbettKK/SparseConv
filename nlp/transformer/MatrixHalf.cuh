//
// Created by dbettkk on 2022/7/23.
//

#ifndef SPARSECONV_MATRIXHALF_CUH
#define SPARSECONV_MATRIXHALF_CUH

#include <cuda_fp16.h>
#include <string>
#include <iostream>
#include <fstream>
#include "kernels_transformer.cuh"


class MatrixHalf {
    half *matrix;
    int batch{}, row{}, col{};

public:
    static void print_device(half *item, int row, int col);

    static void cmp(half *item1, half *item2, int size);

    MatrixHalf(int batch, int row, int col, bool is_device);

    MatrixHalf(int batch, int row, int col, bool is_device, half init);

    MatrixHalf(int batch, int row, int col, bool is_device, std::string path);

    MatrixHalf(half *matrix, int batch, int row, int col);

    void copyTo(MatrixHalf *out);

    half *getMatrix() const;

    void setMatrix(half *matrix);

    int getBatch() const;

    void setBatch(int batch);

    int getRow() const;

    void setRow(int row);

    int getCol() const;

    void setCol(int col);

    int getSize() const;

    void print(const std::string& msg, bool is_device);

    void gemm(MatrixHalf *item, MatrixHalf *out);

    void gemm_batches(MatrixHalf *item, MatrixHalf *out, bool is_single_batch);

    void reshape(MatrixHalf *out, int heads) const;

    void transpose(MatrixHalf *out);

    void softmax();

    void relu();

    void addMatrix(MatrixHalf *add, MatrixHalf *out);

    void free_matrix();

    void layerNorm(MatrixHalf *out);
};


#endif //SPARSECONV_MATRIXHALF_CUH
