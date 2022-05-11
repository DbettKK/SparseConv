import tvm
from tvm import te, topi
import numpy as np


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
        time.append(evaluator(a, b, out).mean * 1e3)

