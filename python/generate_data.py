import random
import torch
import numpy as np

data_path = "../data"


def get_total_size(item):
    mul = 1
    for i in item:
        mul *= i
    return mul


def im2col(input_data: np.ndarray, filter_h, filter_w, stride=1, pad=0, _dtype='float32'):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 输出矩阵的高
    out_w = (W + 2 * pad - filter_w) // stride + 1  # 输出矩阵的宽

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)).astype(_dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def make_dense_data(size: int, bound: int, data_type) -> np.ndarray:
    return np.random.uniform(1, bound, size=size).astype(data_type)


def make_spmma_data(size: int, bound: int, data_type) -> np.ndarray:
    ret = make_dense_data(size, bound, data_type)
    for i in range(size // 2):
        zero_index = random.randint(0, 1)
        ret[i * 2 + zero_index] = 0
    return ret


def make_spmma_data_percent(row: int, col: int, bound: int, percent: int, data_type) -> np.ndarray:
    ret = make_dense_data(row * col, bound, data_type)
    now_percent = 0
    assert percent >= 50
    j_range = col // 2 if col % 2 == 0 else col // 2 + 1
    for i in range(row):
        for j in range(j_range):
            index = i * col + j * 2
            if j * 2 + 1 >= col:
                ret[index] = 0
                now_percent += 1
            else:
                zero_index = random.randint(0, 1)
                ret[index + zero_index] = 0
                now_percent += 1

    non_zero_index_list = []
    for i in range(row * col):
        if ret[i] != 0:
            non_zero_index_list.append(i)
    # 打乱
    for i in range(len(non_zero_index_list)):
        rand_index = random.randint(0, len(non_zero_index_list) - 1)
        tmp = non_zero_index_list[i]
        non_zero_index_list[i] = non_zero_index_list[rand_index]
        non_zero_index_list[rand_index] = tmp
    if now_percent / (row * col) >= percent:
        return ret
    else:
        need_zero = int(row * col * percent / 100) - now_percent
        assert len(non_zero_index_list) >= need_zero
        for i in range(need_zero):
            ret[non_zero_index_list[i]] = 0
        return ret


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


def mma(m: int, k: int, n: int, bound: int):
    A = make_spmma_data(m * k, bound, 'float32').astype(int).astype('float32')
    B = make_dense_data(k * n, bound, 'float32').astype(int).astype('float32')
    C = np.zeros(m * n, 'float32').astype(int).astype('float32')
    print(np.matmul(A.reshape(m, k), B.reshape(k, n)))
    A.tofile(data_path + '/a.bin')
    B.tofile(data_path + '/b.bin')
    C.tofile(data_path + '/c.bin')


def conv(data_size: list, kernel_size: list, padding: int, stride: int, data_bound: int, kernel_bound: int):
    data_size_total = 1
    kernel_size_total = 1
    for i in range(4):
        data_size_total *= data_size[i]
        kernel_size_total *= kernel_size[i]

    data = make_dense_data(data_size_total, data_bound, 'float32')
    kernel = make_sparse_kernel(kernel_size[0], kernel_size_total // kernel_size[0], kernel_bound, 'float32')
    data.tofile(data_path + '/data.bin')
    kernel.tofile(data_path + '/kernel.bin')
    ans = torch.nn.functional.conv2d(torch.from_numpy(data.reshape(data_size)),
                                     torch.from_numpy(kernel.reshape(kernel_size)), stride=1, padding=0)
    print(ans)


def generate_data_filter(d_size, f_size, d_path, f_path, d_bound, f_bound):
    data1 = make_dense_data(get_total_size(d_size), d_bound, 'float16')
    filter1 = make_sparse_kernel(f_size[0], get_total_size(f_size) // f_size[0], f_bound, 'float16')
    data1.tofile(d_path)
    filter1.tofile(f_path)


def generate_data_matmul(d_size, f_size, padding, stride, a_path, b_path, d_bound, f_bound):
    data = make_dense_data(get_total_size(d_size), d_bound, 'float16')
    filter1 = make_sparse_kernel(f_size[0], get_total_size(f_size) // f_size[0], f_bound, 'float16')

    filter1.tofile(a_path)
    col = im2col(data.reshape(d_size), f_size[2], f_size[3], stride, padding, 'float16').T
    col.tofile(b_path)


def generate_data_matmul_percent(d_size, f_size, padding, stride, a_path, b_path, d_bound, f_bound, percent):
    data = make_dense_data(get_total_size(d_size), d_bound, 'float16')
    filter1 = make_spmma_data_percent(f_size[0], get_total_size(f_size) // f_size[0], f_bound, percent, 'float16')
    filter1.tofile(a_path)
    col = im2col(data.reshape(d_size), f_size[2], f_size[3], stride, padding, 'float16').T
    col.tofile(b_path)


def generate_vgg():
    vgg_path = "/vgg"
    generate_data_filter([1, 3, 224, 224], [64, 3, 3, 3], data_path + vgg_path + "/data1.bin",
                         data_path + vgg_path + "/filter1.bin", 10, 8)
    generate_data_filter([1, 64, 224, 224], [64, 64, 3, 3], data_path + vgg_path + "/data2.bin",
                         data_path + vgg_path + "/filter2.bin", 10, 8)
    generate_data_filter([1, 64, 112, 112], [128, 64, 3, 3], data_path + vgg_path + "/data3.bin",
                         data_path + vgg_path + "/filter3.bin", 10, 8)
    generate_data_filter([1, 128, 112, 112], [128, 128, 3, 3], data_path + vgg_path + "/data4.bin",
                         data_path + vgg_path + "/filter4.bin", 10, 8)
    generate_data_filter([1, 128, 56, 56], [256, 128, 3, 3], data_path + vgg_path + "/data5.bin",
                         data_path + vgg_path + "/filter5.bin", 10, 8)
    generate_data_filter([1, 256, 56, 56], [256, 256, 3, 3], data_path + vgg_path + "/data6.bin",
                         data_path + vgg_path + "/filter6.bin", 10, 8)
    generate_data_filter([1, 256, 56, 56], [256, 256, 3, 3], data_path + vgg_path + "/data7.bin",
                         data_path + vgg_path + "/filter7.bin", 10, 8)
    generate_data_filter([1, 256, 28, 28], [512, 256, 3, 3], data_path + vgg_path + "/data8.bin",
                         data_path + vgg_path + "/filter8.bin", 10, 8)
    generate_data_filter([1, 512, 28, 28], [512, 512, 3, 3], data_path + vgg_path + "/data9.bin",
                         data_path + vgg_path + "/filter9.bin", 10, 8)
    generate_data_filter([1, 512, 28, 28], [512, 512, 3, 3], data_path + vgg_path + "/data10.bin",
                         data_path + vgg_path + "/filter10.bin", 10, 8)
    generate_data_filter([1, 512, 14, 14], [512, 512, 3, 3], data_path + vgg_path + "/data11.bin",
                         data_path + vgg_path + "/filter11.bin", 10, 8)
    generate_data_filter([1, 512, 14, 14], [512, 512, 3, 3], data_path + vgg_path + "/data12.bin",
                         data_path + vgg_path + "/filter12.bin", 10, 8)
    generate_data_filter([1, 512, 14, 14], [512, 512, 3, 3], data_path + vgg_path + "/data13.bin",
                         data_path + vgg_path + "/filter13.bin", 10, 8)


def get_vgg_mm():
    print("vgg")
    get_matmul([1, 3, 224, 224], [64, 3, 3, 3], 1, 1)
    get_matmul([1, 64, 224, 224], [64, 64, 3, 3], 1, 1)
    get_matmul([1, 64, 112, 112], [128, 64, 3, 3], 1, 1)
    get_matmul([1, 128, 112, 112], [128, 128, 3, 3], 1, 1)
    get_matmul([1, 128, 56, 56], [256, 128, 3, 3], 1, 1)
    get_matmul([1, 256, 56, 56], [256, 256, 3, 3], 1, 1)
    get_matmul([1, 256, 56, 56], [256, 256, 3, 3], 1, 1)
    get_matmul([1, 256, 28, 28], [512, 256, 3, 3], 1, 1)
    get_matmul([1, 512, 28, 28], [512, 512, 3, 3], 1, 1)
    get_matmul([1, 512, 28, 28], [512, 512, 3, 3], 1, 1)
    get_matmul([1, 512, 14, 14], [512, 512, 3, 3], 1, 1)
    get_matmul([1, 512, 14, 14], [512, 512, 3, 3], 1, 1)
    get_matmul([1, 512, 14, 14], [512, 512, 3, 3], 1, 1)


def get_resnet_mm():
    print("res")
    get_matmul([1, 3, 224, 224], [64, 3, 7, 7], 3, 2)
    get_matmul([1, 64, 56, 56], [64, 64, 3, 3], 1, 1)
    get_matmul([1, 64, 56, 56], [64, 64, 3, 3], 1, 1)
    get_matmul([1, 64, 56, 56], [64, 64, 3, 3], 1, 1)
    get_matmul([1, 64, 56, 56], [64, 64, 3, 3], 1, 1)
    get_matmul([1, 64, 56, 56], [128, 64, 3, 3], 1, 2)
    get_matmul([1, 128, 28, 28], [128, 128, 3, 3], 1, 1)
    get_matmul([1, 128, 28, 28], [128, 128, 3, 3], 1, 1)
    get_matmul([1, 128, 28, 28], [128, 128, 3, 3], 1, 1)
    get_matmul([1, 128, 28, 28], [256, 128, 3, 3], 1, 2)
    get_matmul([1, 256, 14, 14], [256, 256, 3, 3], 1, 1)
    get_matmul([1, 256, 14, 14], [256, 256, 3, 3], 1, 1)
    get_matmul([1, 256, 14, 14], [256, 256, 3, 3], 1, 1)
    get_matmul([1, 256, 14, 14], [512, 256, 3, 3], 1, 2)
    get_matmul([1, 512, 7, 7], [512, 512, 3, 3], 1, 1)
    get_matmul([1, 512, 7, 7], [512, 512, 3, 3], 1, 1)
    get_matmul([1, 512, 7, 7], [512, 512, 3, 3], 1, 1)


def get_alexnet_mm():
    print('alex')
    get_matmul([1, 3, 227, 227], [96, 3, 11, 11], 0, 4)
    get_matmul([1, 96, 27, 27], [256, 96, 5, 5], 2, 1)
    get_matmul([1, 256, 13, 13], [384, 256, 3, 3], 1, 1)
    get_matmul([1, 384, 13, 13], [384, 384, 3, 3], 1, 1)
    get_matmul([1, 384, 13, 13], [256, 384, 3, 3], 1, 1)


def get_matmul(data_size, filter_size, padding, stride):
    m = filter_size[0]
    k = get_total_size(filter_size) // filter_size[0]
    out_h = (data_size[2] + 2 * padding - filter_size[2]) // stride + 1
    out_w = (data_size[3] + 2 * padding - filter_size[3]) // stride + 1
    n = data_size[0] * out_h * out_w
    print("m%dk%dn%d" % (m, k, n))


def generate_vgg_matmul():
    vgg_path = "/vgg"
    generate_data_matmul([1, 3, 224, 224], [64, 3, 3, 3], 1, 1, data_path + vgg_path + "/a1.bin",
                         data_path + vgg_path + "/b1.bin", 10, 8)
    generate_data_matmul([1, 64, 224, 224], [64, 64, 3, 3], 1, 1, data_path + vgg_path + "/a2.bin",
                         data_path + vgg_path + "/b2.bin", 10, 8)
    generate_data_matmul([1, 64, 112, 112], [128, 64, 3, 3], 1, 1, data_path + vgg_path + "/a3.bin",
                         data_path + vgg_path + "/b3.bin", 10, 8)
    generate_data_matmul([1, 128, 112, 112], [128, 128, 3, 3], 1, 1, data_path + vgg_path + "/a4.bin",
                         data_path + vgg_path + "/b4.bin", 10, 8)
    generate_data_matmul([1, 128, 56, 56], [256, 128, 3, 3], 1, 1, data_path + vgg_path + "/a5.bin",
                         data_path + vgg_path + "/b5.bin", 10, 8)
    generate_data_matmul([1, 256, 56, 56], [256, 256, 3, 3], 1, 1, data_path + vgg_path + "/a6.bin",
                         data_path + vgg_path + "/b6.bin", 10, 8)
    generate_data_matmul([1, 256, 56, 56], [256, 256, 3, 3], 1, 1, data_path + vgg_path + "/a7.bin",
                         data_path + vgg_path + "/b7.bin", 10, 8)
    generate_data_matmul([1, 256, 28, 28], [512, 256, 3, 3], 1, 1, data_path + vgg_path + "/a8.bin",
                         data_path + vgg_path + "/b8.bin", 10, 8)
    generate_data_matmul([1, 512, 28, 28], [512, 512, 3, 3], 1, 1, data_path + vgg_path + "/a9.bin",
                         data_path + vgg_path + "/b9.bin", 10, 8)
    generate_data_matmul([1, 512, 28, 28], [512, 512, 3, 3], 1, 1, data_path + vgg_path + "/a10.bin",
                         data_path + vgg_path + "/b10.bin", 10, 8)
    generate_data_matmul([1, 512, 14, 14], [512, 512, 3, 3], 1, 1, data_path + vgg_path + "/a11.bin",
                         data_path + vgg_path + "/b11.bin", 10, 8)
    generate_data_matmul([1, 512, 14, 14], [512, 512, 3, 3], 1, 1, data_path + vgg_path + "/a12.bin",
                         data_path + vgg_path + "/b12.bin", 10, 8)
    generate_data_matmul([1, 512, 14, 14], [512, 512, 3, 3], 1, 1, data_path + vgg_path + "/a13.bin",
                         data_path + vgg_path + "/b13.bin", 10, 8)


def generate_resnet18():
    resnet_path = "/resnet"
    generate_data_filter([16, 3, 224, 224], [64, 3, 7, 7], data_path + resnet_path + "/data1.bin",
                         data_path + resnet_path + "/filter1.bin", 10, 8)
    generate_data_filter([16, 64, 56, 56], [64, 64, 3, 3], data_path + resnet_path + "/data2.bin",
                         data_path + resnet_path + "/filter2.bin", 10, 8)
    generate_data_filter([16, 64, 56, 56], [64, 64, 3, 3], data_path + resnet_path + "/data3.bin",
                         data_path + resnet_path + "/filter3.bin", 10, 8)
    generate_data_filter([16, 64, 56, 56], [64, 64, 3, 3], data_path + resnet_path + "/data4.bin",
                         data_path + resnet_path + "/filter4.bin", 10, 8)
    generate_data_filter([16, 64, 56, 56], [64, 64, 3, 3], data_path + resnet_path + "/data5.bin",
                         data_path + resnet_path + "/filter5.bin", 10, 8)
    generate_data_filter([16, 64, 56, 56], [128, 64, 3, 3], data_path + resnet_path + "/data6.bin",
                         data_path + resnet_path + "/filter6.bin", 10, 8)
    generate_data_filter([16, 128, 28, 28], [128, 128, 3, 3], data_path + resnet_path + "/data7.bin",
                         data_path + resnet_path + "/filter7.bin", 10, 8)
    generate_data_filter([16, 128, 28, 28], [128, 128, 3, 3], data_path + resnet_path + "/data8.bin",
                         data_path + resnet_path + "/filter8.bin", 10, 8)
    generate_data_filter([16, 128, 28, 28], [128, 128, 3, 3], data_path + resnet_path + "/data9.bin",
                         data_path + resnet_path + "/filter9.bin", 10, 8)
    generate_data_filter([16, 128, 28, 28], [256, 128, 3, 3], data_path + resnet_path + "/data10.bin",
                         data_path + resnet_path + "/filter10.bin", 10, 8)
    generate_data_filter([16, 256, 14, 14], [256, 256, 3, 3], data_path + resnet_path + "/data11.bin",
                         data_path + resnet_path + "/filter11.bin", 10, 8)
    generate_data_filter([16, 256, 14, 14], [256, 256, 3, 3], data_path + resnet_path + "/data12.bin",
                         data_path + resnet_path + "/filter12.bin", 10, 8)
    generate_data_filter([16, 256, 14, 14], [256, 256, 3, 3], data_path + resnet_path + "/data13.bin",
                         data_path + resnet_path + "/filter13.bin", 10, 8)
    generate_data_filter([16, 256, 14, 14], [512, 256, 3, 3], data_path + resnet_path + "/data14.bin",
                         data_path + resnet_path + "/filter14.bin", 10, 8)
    generate_data_filter([16, 512, 7, 7], [512, 512, 3, 3], data_path + resnet_path + "/data15.bin",
                         data_path + resnet_path + "/filter15.bin", 10, 8)
    generate_data_filter([16, 512, 7, 7], [512, 512, 3, 3], data_path + resnet_path + "/data16.bin",
                         data_path + resnet_path + "/filter16.bin", 10, 8)
    generate_data_filter([16, 512, 7, 7], [512, 512, 3, 3], data_path + resnet_path + "/data17.bin",
                         data_path + resnet_path + "/filter17.bin", 10, 8)


def generate_resnet50():
    resnet_path = "/resnet50"
    generate_data_filter([1, 3, 224, 224], [64, 3, 7, 7], data_path + resnet_path + "/data1.bin",
                         data_path + resnet_path + "/filter1.bin", 10, 8)
    # conv_2
    generate_data_filter([1, 64, 56, 56], [64, 64, 1, 1], data_path + resnet_path + "/data2.bin",
                         data_path + resnet_path + "/filter2.bin", 10, 8)
    generate_data_filter([1, 64, 56, 56], [64, 64, 3, 3], data_path + resnet_path + "/data3.bin",
                         data_path + resnet_path + "/filter3.bin", 10, 8)
    generate_data_filter([1, 64, 56, 56], [256, 64, 1, 1], data_path + resnet_path + "/data4.bin",
                         data_path + resnet_path + "/filter4.bin", 10, 8)
    for i in range(2):
        generate_data_filter([1, 256, 56, 56], [64, 256, 1, 1],
                             data_path + resnet_path + "/data" + str(5 + i * 3) + ".bin",
                             data_path + resnet_path + "/filter" + str(5 + i * 3) + ".bin", 10, 8)
        generate_data_filter([1, 64, 56, 56], [64, 64, 3, 3],
                             data_path + resnet_path + "/data" + str(6 + i * 3) + ".bin",
                             data_path + resnet_path + "/filter" + str(6 + i * 3) + ".bin", 10, 8)
        generate_data_filter([1, 64, 56, 56], [256, 64, 1, 1],
                             data_path + resnet_path + "/data" + str(7 + i * 3) + ".bin",
                             data_path + resnet_path + "/filter" + str(7 + i * 3) + ".bin", 10, 8)
    # conv_3
    generate_data_filter([1, 256, 56, 56], [128, 256, 1, 1], data_path + resnet_path + "/data11.bin",
                         data_path + resnet_path + "/filter11.bin", 10, 8)
    generate_data_filter([1, 128, 28, 28], [128, 128, 3, 3], data_path + resnet_path + "/data12.bin",
                         data_path + resnet_path + "/filter12.bin", 10, 8)
    generate_data_filter([1, 128, 28, 28], [512, 128, 1, 1], data_path + resnet_path + "/data13.bin",
                         data_path + resnet_path + "/filter13.bin", 10, 8)
    for i in range(3):
        generate_data_filter([1, 512, 28, 28], [128, 512, 1, 1],
                             data_path + resnet_path + "/data" + str(14 + i * 3) + ".bin",
                             data_path + resnet_path + "/filter" + str(14 + i * 3) + ".bin", 10, 8)
        generate_data_filter([1, 128, 28, 28], [128, 128, 3, 3],
                             data_path + resnet_path + "/data" + str(15 + i * 3) + ".bin",
                             data_path + resnet_path + "/filter" + str(15 + i * 3) + ".bin", 10, 8)
        generate_data_filter([1, 128, 28, 28], [512, 128, 1, 1],
                             data_path + resnet_path + "/data" + str(16 + i * 3) + ".bin",
                             data_path + resnet_path + "/filter" + str(16 + i * 3) + ".bin", 10, 8)

    # conv_4
    generate_data_filter([1, 512, 28, 28], [256, 512, 1, 1], data_path + resnet_path + "/data23.bin",
                         data_path + resnet_path + "/filter23.bin", 10, 8)
    generate_data_filter([1, 256, 14, 14], [256, 256, 3, 3], data_path + resnet_path + "/data24.bin",
                         data_path + resnet_path + "/filter24.bin", 10, 8)
    generate_data_filter([1, 256, 14, 14], [1024, 256, 1, 1], data_path + resnet_path + "/data25.bin",
                         data_path + resnet_path + "/filter25.bin", 10, 8)
    for i in range(5):
        generate_data_filter([1, 1024, 14, 14], [256, 1024, 1, 1],
                             data_path + resnet_path + "/data" + str(26 + i * 3) + ".bin",
                             data_path + resnet_path + "/filter" + str(26 + i * 3) + ".bin", 10, 8)
        generate_data_filter([1, 256, 14, 14], [256, 256, 3, 3],
                             data_path + resnet_path + "/data" + str(27 + i * 3) + ".bin",
                             data_path + resnet_path + "/filter" + str(27 + i * 3) + ".bin", 10, 8)
        generate_data_filter([1, 256, 14, 14], [1024, 256, 1, 1],
                             data_path + resnet_path + "/data" + str(28 + i * 3) + ".bin",
                             data_path + resnet_path + "/filter" + str(28 + i * 3) + ".bin", 10, 8)

    # conv_5
    generate_data_filter([1, 1024, 14, 14], [512, 1024, 1, 1], data_path + resnet_path + "/data41.bin",
                         data_path + resnet_path + "/filter41.bin", 10, 8)
    generate_data_filter([1, 512, 7, 7], [512, 512, 3, 3], data_path + resnet_path + "/data42.bin",
                         data_path + resnet_path + "/filter42.bin", 10, 8)
    generate_data_filter([1, 512, 7, 7], [2048, 512, 1, 1], data_path + resnet_path + "/data43.bin",
                         data_path + resnet_path + "/filter43.bin", 10, 8)
    for i in range(2):
        generate_data_filter([1, 2048, 7, 7], [512, 2048, 1, 1],
                             data_path + resnet_path + "/data" + str(44 + i * 3) + ".bin",
                             data_path + resnet_path + "/filter" + str(44 + i * 3) + ".bin", 10, 8)
        generate_data_filter([1, 512, 7, 7], [512, 512, 3, 3],
                             data_path + resnet_path + "/data" + str(45 + i * 3) + ".bin",
                             data_path + resnet_path + "/filter" + str(45 + i * 3) + ".bin", 10, 8)
        generate_data_filter([1, 512, 7, 7], [2048, 512, 1, 1],
                             data_path + resnet_path + "/data" + str(46 + i * 3) + ".bin",
                             data_path + resnet_path + "/filter" + str(46 + i * 3) + ".bin", 10, 8)


def generate_resnet18_matmul():
    resnet_path = "/resnet"
    generate_data_matmul([1, 3, 224, 224], [64, 3, 7, 7], 3, 2, data_path + resnet_path + "/a1.bin",
                         data_path + resnet_path + "/b1.bin", 10, 8)
    generate_data_matmul([1, 64, 56, 56], [64, 64, 3, 3], 1, 1, data_path + resnet_path + "/a2.bin",
                         data_path + resnet_path + "/b2.bin", 10, 8)
    generate_data_matmul([1, 64, 56, 56], [64, 64, 3, 3], 1, 1, data_path + resnet_path + "/a3.bin",
                         data_path + resnet_path + "/b3.bin", 10, 8)
    generate_data_matmul([1, 64, 56, 56], [64, 64, 3, 3], 1, 1, data_path + resnet_path + "/a4.bin",
                         data_path + resnet_path + "/b4.bin", 10, 8)
    generate_data_matmul([1, 64, 56, 56], [64, 64, 3, 3], 1, 1, data_path + resnet_path + "/a5.bin",
                         data_path + resnet_path + "/b5.bin", 10, 8)
    generate_data_matmul([1, 64, 56, 56], [128, 64, 3, 3], 1, 2, data_path + resnet_path + "/a6.bin",
                         data_path + resnet_path + "/b6.bin", 10, 8)
    generate_data_matmul([1, 128, 28, 28], [128, 128, 3, 3], 1, 1, data_path + resnet_path + "/a7.bin",
                         data_path + resnet_path + "/b7.bin", 10, 8)
    generate_data_matmul([1, 128, 28, 28], [128, 128, 3, 3], 1, 1, data_path + resnet_path + "/a8.bin",
                         data_path + resnet_path + "/b8.bin", 10, 8)
    generate_data_matmul([1, 128, 28, 28], [128, 128, 3, 3], 1, 1, data_path + resnet_path + "/a9.bin",
                         data_path + resnet_path + "/b9.bin", 10, 8)
    generate_data_matmul([1, 128, 28, 28], [256, 128, 3, 3], 1, 2, data_path + resnet_path + "/a10.bin",
                         data_path + resnet_path + "/b10.bin", 10, 8)
    generate_data_matmul([1, 256, 14, 14], [256, 256, 3, 3], 1, 1, data_path + resnet_path + "/a11.bin",
                         data_path + resnet_path + "/b11.bin", 10, 8)
    generate_data_matmul([1, 256, 14, 14], [256, 256, 3, 3], 1, 1, data_path + resnet_path + "/a12.bin",
                         data_path + resnet_path + "/b12.bin", 10, 8)
    generate_data_matmul([1, 256, 14, 14], [256, 256, 3, 3], 1, 1, data_path + resnet_path + "/a13.bin",
                         data_path + resnet_path + "/b13.bin", 10, 8)
    generate_data_matmul([1, 256, 14, 14], [512, 256, 3, 3], 1, 2, data_path + resnet_path + "/a14.bin",
                         data_path + resnet_path + "/b14.bin", 10, 8)
    generate_data_matmul([1, 512, 7, 7], [512, 512, 3, 3], 1, 1, data_path + resnet_path + "/a15.bin",
                         data_path + resnet_path + "/b15.bin", 10, 8)
    generate_data_matmul([1, 512, 7, 7], [512, 512, 3, 3], 1, 1, data_path + resnet_path + "/a16.bin",
                         data_path + resnet_path + "/b16.bin", 10, 8)
    generate_data_matmul([1, 512, 7, 7], [512, 512, 3, 3], 1, 1, data_path + resnet_path + "/a17.bin",
                         data_path + resnet_path + "/b17.bin", 10, 8)


def generate_alexnet():
    alexnet_path = "/alex"
    generate_data_filter([1, 3, 227, 227], [96, 3, 11, 11], data_path + alexnet_path + "/data1.bin",
                         data_path + alexnet_path + "/filter1.bin", 10, 8)
    generate_data_filter([1, 96, 27, 27], [256, 96, 5, 5], data_path + alexnet_path + "/data2.bin",
                         data_path + alexnet_path + "/filter2.bin", 10, 8)
    generate_data_filter([1, 256, 13, 13], [384, 256, 3, 3], data_path + alexnet_path + "/data3.bin",
                         data_path + alexnet_path + "/filter3.bin", 10, 8)
    generate_data_filter([1, 384, 13, 13], [384, 384, 3, 3], data_path + alexnet_path + "/data4.bin",
                         data_path + alexnet_path + "/filter4.bin", 10, 8)
    generate_data_filter([1, 384, 13, 13], [256, 384, 3, 3], data_path + alexnet_path + "/data5.bin",
                         data_path + alexnet_path + "/filter5.bin", 10, 8)


def generate_alexnet_matmul():
    alexnet_path = "/alex"
    generate_data_matmul([1, 3, 227, 227], [96, 3, 11, 11], 0, 4, data_path + alexnet_path + "/a1.bin",
                         data_path + alexnet_path + "/b1.bin", 10, 8)
    generate_data_matmul([1, 96, 27, 27], [256, 96, 5, 5], 2, 1, data_path + alexnet_path + "/a2.bin",
                         data_path + alexnet_path + "/b2.bin", 10, 8)
    generate_data_matmul([1, 256, 13, 13], [384, 256, 3, 3], 1, 1, data_path + alexnet_path + "/a3.bin",
                         data_path + alexnet_path + "/b3.bin", 10, 8)
    generate_data_matmul([1, 384, 13, 13], [384, 384, 3, 3], 1, 1, data_path + alexnet_path + "/a4.bin",
                         data_path + alexnet_path + "/b4.bin", 10, 8)
    generate_data_matmul([1, 384, 13, 13], [256, 384, 3, 3], 1, 1, data_path + alexnet_path + "/a5.bin",
                         data_path + alexnet_path + "/b5.bin", 10, 8)


def generate_alexnet_matmul_percent(percent):
    alexnet_path = "/alex/sparse/" + str(percent)
    generate_data_matmul_percent([1, 3, 227, 227], [96, 3, 11, 11], 0, 4, data_path + alexnet_path + "a1.bin",
                                 data_path + alexnet_path + "b1.bin", 10, 8, percent)
    generate_data_matmul_percent([1, 96, 27, 27], [256, 96, 5, 5], 2, 1, data_path + alexnet_path + "a2.bin",
                                 data_path + alexnet_path + "b2.bin", 10, 8, percent)
    generate_data_matmul_percent([1, 256, 13, 13], [384, 256, 3, 3], 1, 1, data_path + alexnet_path + "a3.bin",
                                 data_path + alexnet_path + "b3.bin", 10, 8, percent)
    generate_data_matmul_percent([1, 384, 13, 13], [384, 384, 3, 3], 1, 1, data_path + alexnet_path + "a4.bin",
                                 data_path + alexnet_path + "b4.bin", 10, 8, percent)
    generate_data_matmul_percent([1, 384, 13, 13], [256, 384, 3, 3], 1, 1, data_path + alexnet_path + "a5.bin",
                                 data_path + alexnet_path + "b5.bin", 10, 8, percent)


def generate_alexnet_percent():
    generate_alexnet_matmul_percent(50)
    generate_alexnet_matmul_percent(55)
    generate_alexnet_matmul_percent(60)
    generate_alexnet_matmul_percent(65)
    generate_alexnet_matmul_percent(70)
    generate_alexnet_matmul_percent(75)
    generate_alexnet_matmul_percent(80)
    generate_alexnet_matmul_percent(85)
    generate_alexnet_matmul_percent(90)
    generate_alexnet_matmul_percent(95)


if __name__ == '__main__':
    # mma(256, 256, 256, 10)
    # conv([4, 3, 16, 16], [4, 3, 3, 3], 0, 1, 10, 10)
    # generate_vgg()
    # resnet_path = "/resnet"
    # generate_data_matmul([1, 3, 224, 224], [64, 3, 7, 7], 3, 2, data_path + resnet_path + "/a1.bin", data_path + resnet_path + "/b1.bin", 10, 8)
    # generate_resnet18()
    # generate_alexnet()
    # generate_vgg_matmul()
    # generate_resnet18_matmul()
    # generate_alexnet_matmul()
    # generate_alexnet_percent()
    generate_resnet18()