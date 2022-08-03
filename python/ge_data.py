import numpy as np
import random


def get_total_size(item):
    mul = 1
    for i in item:
        mul *= i
    return mul


def make_dense_data(size: int, bound: int, data_type) -> np.ndarray:
    return np.random.uniform(0, bound, size=size).astype(data_type)


def make_sparse_kernel(im_n: int, im_k: int, bound: int, data_type) -> np.ndarray:
    ret = make_dense_data(im_n * im_k, bound, data_type)
    j_range = im_k // 4 + 1 if im_k % 4 != 0 else im_k // 4
    for i in range(im_n):
        for j in range(j_range):
            index = i * im_k + j * 4
            if j * 4 + 1 >= im_k:
                ret[index] = 0
            elif j * 4 + 2 >= im_k:
                zero_index = random.randint(0, 1)
                ret[index + zero_index] = 0
            elif j * 4 + 3 >= im_k:
                zero_index = random.randint(0, 1)
                ret[index + zero_index] = 0
                ret[index + 2] = 0
            else:
                for k in range(2):
                    k_index = index + k * 2
                    zero_index = random.randint(0, 1)
                    ret[k_index + zero_index] = 0
    return ret


def ge_filter(kernel_size, conv_num, path):
    filter = make_sparse_kernel(kernel_size[0], get_total_size(kernel_size) // kernel_size[0], 1, 'float16')
    filter.tofile(path + '/filter' + str(conv_num))

def resnet50():
    data_path = '../data/resnet50'
    data_size = [1, 3, 224, 224]
    kernel_size_x = [
        [256, 64, 1, 1],
        [256, 64, 1, 1],
        [256, 64, 1, 1],
        [512, 256, 1, 1],
        [512, 256, 1, 1],
        [512, 256, 1, 1],
        [512, 256, 1, 1],
        [1024, 512, 1, 1],
        [1024, 512, 1, 1],
        [1024, 512, 1, 1],
        [1024, 512, 1, 1],
        [1024, 512, 1, 1],
        [1024, 512, 1, 1],
        [2048, 1024, 1, 1],
        [2048, 1024, 1, 1],
        [2048, 1024, 1, 1],
    ]
    kernel_sizes = [
        [64, 3, 7, 7],
        [64, 64, 1, 1],
        [64, 64, 3, 3],
        [256, 64, 1, 1],
        [64, 256, 1, 1],
        [64, 64, 3, 3],
        [256, 64, 1, 1],
        [64, 256, 1, 1],
        [64, 64, 3, 3],
        [256, 64, 1, 1],
        [128, 256, 1, 1],
        [128, 128, 3, 3],
        [512, 128, 1, 1],
        [128, 512, 1, 1],
        [128, 128, 3, 3],
        [512, 128, 1, 1],
        [128, 512, 1, 1],
        [128, 128, 3, 3],
        [512, 128, 1, 1],
        [128, 512, 1, 1],
        [128, 128, 3, 3],
        [512, 128, 1, 1],
        [256, 512, 1, 1],
        [256, 256, 3, 3],
        [1024, 256, 1, 1],
        [256, 1024, 1, 1],
        [256, 256, 3, 3],
        [1024, 256, 1, 1],
        [256, 1024, 1, 1],
        [256, 256, 3, 3],
        [1024, 256, 1, 1],
        [256, 1024, 1, 1],
        [256, 256, 3, 3],
        [1024, 256, 1, 1],
        [256, 1024, 1, 1],
        [256, 256, 3, 3],
        [1024, 256, 1, 1],
        [256, 1024, 1, 1],
        [256, 256, 3, 3],
        [1024, 256, 1, 1],
        [512, 1024, 1, 1],
        [512, 512, 3, 3],
        [2048, 512, 1, 1],
        [512, 2048, 1, 1],
        [512, 512, 3, 3],
        [2048, 512, 1, 1],
        [512, 2048, 1, 1],
        [512, 512, 3, 3],
        [2048, 512, 1, 1],
    ]
    make_dense_data(get_total_size(data_size), 1, 'float16').tofile(data_path + '/data')
    for i in range(49):
        ge_filter(kernel_sizes[i], i, data_path)
    for i in range(16):
        ge_filter(kernel_size_x[i], 100 + i, data_path)


if __name__ == '__main__':
    resnet50()