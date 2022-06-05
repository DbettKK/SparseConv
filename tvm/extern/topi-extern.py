import ctypes as c
import sys
import numpy as np
import random
import tvm
from tvm import te


def make_dense_data(size: int, bound: int, data_type) -> np.ndarray:
    return np.random.uniform(0, bound, size=size).astype(data_type)


def make_spmma_data(size: int, bound: int, data_type) -> np.ndarray:
    ret = make_dense_data(size, bound, data_type)
    for i in range(size // 2):
        zero_index = random.randint(0, 1)
        ret[i * 2 + zero_index] = 0
    return ret


def make_sparse_kernel(im_n: int, im_k: int, bound: int, data_type) -> np.ndarray:
    ret = make_dense_data(im_n * im_k, bound, data_type)
    for i in range(im_n):
        for j in range(im_k // 2):
            index = i * im_k + j * 2
            if index + 1 >= im_k * im_n:
                ret[index] = 0
            else:
                zero_index = random.randint(0, 1)
                ret[index + zero_index] = 0
    return ret


def get_total_size(item):
    mul = 1
    for i in item:
        mul *= i
    return mul


@tvm.register_func("tvm.contrib.my_topi_conv2d")
def spmma_conv(Image, Kernel, ConvOut):
    libc = c.cdll.LoadLibrary('../../spmma/tmp.so')

    data_size = Image.shape
    kernel_size = Kernel.shape
    out_size = ConvOut.shape
    data_size_total = get_total_size(data_size)
    kernel_size_total = get_total_size(kernel_size)
    out_size_total = get_total_size(out_size)

    image_data = Image.numpy().reshape(data_size_total)
    kernel_data = Kernel.numpy().reshape(kernel_size_total)
    out_data = ConvOut.numpy().reshape(out_size_total)

    Data = (c.c_float*len(image_data))(*image_data)
    Kernel = (c.c_float*len(kernel_data))(*kernel_data)
    Out = (c.c_float*len(out_data))(*out_data)

    libc.conv2d(Data, Kernel, Out)

    tvm.nd.array(np.frombuffer(Out, 'float32').reshape(out_size)).copyto(ConvOut)


def tvm_call_conv2d():
    data_size = [16, 16, 7, 7]
    kernel_size = [16, 16, 4, 4]
    kernel_size_total = get_total_size(kernel_size)
    out_n = data_size[0]
    out_c = kernel_size[0]
    out_h = data_size[2] - kernel_size[2] + 1
    out_w = data_size[3] - kernel_size[3] + 1
    out_size = [out_n, out_c, out_h, out_w]

    Image = te.placeholder(data_size, name='Image')
    Filter = te.placeholder(kernel_size, name='Kernel')
    Out = te.extern(
        out_size,
        [Image, Filter],
        lambda ins, outs: tvm.tir.call_packed("tvm.contrib.my_topi_conv2d", ins[0], ins[1], outs[0]),
        name='Out',
    )

    s = te.create_schedule(Out.op)
    f = tvm.build(s, [Image, Filter, Out], "cuda")

    image_data = np.random.uniform(1, 10, size=data_size).astype('float32')
    kernel_data = make_sparse_kernel(kernel_size[0], kernel_size_total // kernel_size[0], 10, 'float32').reshape(kernel_size)

    a = tvm.nd.array(image_data, tvm.cuda(0))
    b = tvm.nd.array(kernel_data, tvm.cuda(0))
    out = tvm.nd.array(np.zeros((out_n, out_c, out_h, out_w), Out.dtype), tvm.cuda(0))
    f(a, b, out)

    # print(out.numpy())
    # print(tvm.lower(s, [Image, Filter, Out]))
    evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=10)
    print("Convolution: %f ms" % (evaluator(a, b, out).mean * 1e3))


if __name__ == '__main__':
    tvm_call_conv2d()


