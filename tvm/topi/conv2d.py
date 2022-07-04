import tvm
from tvm import te, topi
import numpy as np
import ctypes

# import torch


time = []


def get_total_size(item):
    mul = 1
    for i in item.shape:
        mul *= i
    return mul


def check_correct(item1: np.ndarray, item2: np.ndarray, ac):
    assert get_total_size(item1) == get_total_size(item2)
    total = get_total_size(item1)
    item1 = item1.reshape(total)
    item2 = item2.reshape(total)
    diff = 0
    for i in range(get_total_size(item1)):
        if abs(item1[i] - item2[i]) > ac:
            diff += 1
    print("total: %d, diff: %d" % (total, diff))


def topi_cuda(data_size: list, filter_size: list, padding: int, stride: int, d_p: str, f_p: str):
    global time
    with tvm.target.Target("cuda"):
        Data = te.placeholder(data_size, name='Data', dtype='float32')
        Filter = te.placeholder(filter_size, name='Filter', dtype='float32')
        Out = topi.cuda.conv2d_nchw(Data, Filter, stride, padding, 1)

        s = topi.cuda.schedule_conv2d_nchw([Out])
        f = tvm.build(s, [Data, Filter, Out], "cuda")

        # image_data = np.random.uniform(1, 10, size=data_size).astype('float32')
        # kernel_data = np.random.uniform(1, 8, size=filter_size).astype('float32')

        a = tvm.nd.array(np.fromfile(d_p, 'float32').reshape(data_size), tvm.cuda(0))
        b = tvm.nd.array(np.fromfile(f_p, 'float32').reshape(filter_size), tvm.cuda(0))
        out = tvm.nd.array(np.zeros([int(i) for i in Out.shape], Out.dtype), tvm.cuda(0))

        f(a, b, out)
        # torch不支持float16
        # ans = torch.nn.functional.conv2d(
        #   torch.from_numpy(image_data), torch.from_numpy(kernel_data), stride=stride, padding=padding)
        # check_correct(out.numpy(), ans.numpy(), 0.01)
        evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=1)
        print("Convolution: %f ms" % (evaluator(a, b, out).mean * 1e3))
        print(f.imported_modules[0].get_source())
        time.append(evaluator(a, b, out).mean * 1e3)


def topi_conv2d(data_size: list, filter_size: list, padding, stride):
    with tvm.target.Target("cuda"):
        Data = te.placeholder(data_size, name='Data', dtype='float16')
        Filter = te.placeholder(filter_size, name='Filter', dtype='float16')
        Out = topi.cuda.conv2d_nhwc_tensorcore(Data, Filter, stride, padding, 1, 'float16')

        s = topi.cuda.schedule_conv2d_nhwc_tensorcore([Out])

        f = tvm.build(s, [Data, Filter, Out], "cuda")

        image_data = np.random.uniform(1, 10, size=data_size).astype('float16')
        kernel_data = np.random.uniform(1, 8, size=filter_size).astype('float16')

        a = tvm.nd.array(image_data.reshape(data_size), tvm.cuda(0))
        b = tvm.nd.array(kernel_data.reshape(filter_size), tvm.cuda(0))
        out = tvm.nd.array(np.zeros([int(i) for i in Out.shape], Out.dtype), tvm.cuda(0))

        f(a, b, out)

        # torch不支持float16
        # ans = torch.nn.functional.conv2d(
        #   torch.from_numpy(image_data), torch.from_numpy(kernel_data), stride=stride, padding=padding)
        # check_correct(out.numpy(), ans.numpy(), 0.01)

        # evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=10)
        # print("Convolution: %f ms" % (evaluator(a, b, out).mean * 1e3))
        # with open("./cuda/wmma.cu", 'w') as out:
        #     out.write(f.imported_modules[0].get_source())
        # tvm.lower(s, [Data, Filter, Out])
        with open('./tmp.txt', 'a') as fi:
            fi.write(str(f.imported_modules[0].get_source()))
            fi.write("\n=========================================\n")
        with open('./tmp2.txt', 'a') as fi:
            fi.write(str(tvm.lower(s, [Data, Filter, Out])))
            fi.write("\n=========================================\n")
        # print(f.imported_modules[0].get_source())
        # print(evaluator(a, b, out).mean * 1e3)


def get_data():
    # batch in_c d_h d_w, out_c in_c f_w f_h, pad stride
    data = [
        [16, 64, 56, 56, 64, 64, 3, 3, 1, 1],
        [16, 64, 56, 56, 64, 64, 3, 3, 1, 1],
        [16, 64, 56, 56, 64, 64, 3, 3, 1, 1],
        [16, 64, 56, 56, 64, 64, 3, 3, 1, 1],
        [16, 64, 56, 56, 128, 64, 3, 3, 1, 2],
        [16, 128, 28, 28, 128, 128, 3, 3, 1, 1],
        [16, 128, 28, 28, 128, 128, 3, 3, 1, 1],
        [16, 128, 28, 28, 128, 128, 3, 3, 1, 1],
        [16, 128, 28, 28, 256, 128, 3, 3, 1, 2],
        [16, 256, 14, 14, 256, 256, 3, 3, 1, 1],
        [16, 256, 14, 14, 256, 256, 3, 3, 1, 1],
        [16, 256, 14, 14, 256, 256, 3, 3, 1, 1],
        [16, 256, 14, 14, 512, 256, 3, 3, 1, 2],
        [16, 512, 7, 7, 512, 512, 3, 3, 1, 1],
        [16, 512, 7, 7, 512, 512, 3, 3, 1, 1],
        [16, 512, 7, 7, 512, 512, 3, 3, 1, 1]
    ]
    datas = []
    filters = []
    paddings = []
    strides = []
    for d in data:
        datas.append([d[0], d[2], d[3], d[1]])
        filters.append([d[6], d[7], d[5], d[4]])
        paddings.append(d[8])
        strides.append(d[9])
    # print(datas, filters, paddings, strides)
    return datas, filters, paddings, strides


if __name__ == '__main__':
    # RTX3070-Windows
    # topi: 0.01ms
    # spmma: gemm-0.02ms / total-0.3ms
    # The shape of (batch, in_channel, num_filter) must be multiple of (16, 16, 16) or (32, 16, 8) or (8, 16, 32) for now
    # A100
    # topi: 0.005306 ms
    # spmma:

    datas, filters, paddings, strides = get_data()

    for i in range(len(datas)):
        topi_conv2d(datas[i], filters[i], paddings[i], strides[i])

    # _data_size = [16, 56, 56, 64]  # nhwc
    # _filter_size = [3, 3, 64, 64]  # hwio
    # _padding = [1, 1]
    # _stride = [1, 1]
    #
    # topi_conv2d(_data_size, _filter_size, _padding, _stride)
