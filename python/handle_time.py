import re
import os


def handle(path: str):
    contents = []
    layer_time = {}
    layer_avg_time = {}
    with open(path, 'r') as f:
        contents = f.readlines()
    for content in contents:
        layer = re.findall(r"conv(.+?):", content)[0]
        time = float(re.findall(r": (.+?)ms", content)[0])
        if layer_time.keys().__contains__(layer) is not True:
            layer_time[layer] = []
        layer_time[layer].append(time)
        # print(layer_time)
    for layer, time in layer_time.items():
        time.sort()
        time.pop(len(time) - 1)
        time.pop(0)
    for layer, time in layer_time.items():
        sum_time = 0.0
        for t in time:
            sum_time += t
        layer_avg_time[layer] = sum_time / len(time)
    for layer, time in layer_avg_time.items():
        print('layer%s,%.5f' % (layer, time))


def handle2(path: str):
    types = [r"spmma: (.+?)ms", r"spmma-noPad: (.+?)ms", r"cublas: (.+?)ms", r"cusparse-csr: (.+?)ms", r"cusparse-coo"
                                                                                                       r": (.+?)ms"]
    sum = [0.0, 0.0, 0.0, 0.0, 0.0]
    with open(path, 'r') as f:
        contents = f.readlines()
    for content in contents:
        for idx, t in enumerate(types):
            if len(re.findall(t, content)) > 0:
                # print(re.findall(t, content))
                sum[idx] += float(re.findall(t, content)[0])
    for i, s in enumerate(sum):
        # batch=2 print(types[i], ":", s / 287, "ms")
        print(types[i], ":", s / 2303, "ms")


def handle_9_1(path: str):
    layer_time = {}
    layer_avg_time = {}
    with open(path, 'r') as f:
        contents = f.readlines()
    for content in contents:
        layer = re.findall(r"conv(.+?):", content)[0]
        time = float(re.findall(r": (.+?)ms", content)[0])
        if layer_time.keys().__contains__(layer) is not True:
            layer_time[layer] = []
        layer_time[layer].append(time)
        # print(layer_time)
    for layer, time in layer_time.items():
        time.sort()
        time.pop(len(time) - 1)
        time.pop(0)
    for layer, time in layer_time.items():
        sum_time = 0.0
        for t in time:
            sum_time += t
        layer_avg_time[layer] = sum_time / len(time)
    for layer, time in layer_avg_time.items():
        # print('layer%s,%.5f' % (layer, time))
        print('conv%s' % layer)


def handle3(path: str):
    with open(path, 'r') as f:
        contents = f.readlines()
    total = 0.0
    for i, content in enumerate(contents):
        if i != 0:
            time = float(content)
            total += time
    print("times: ", len(contents) - 1)
    print(total / (len(contents) - 1))
    os.remove(path)


if __name__ == '__main__':
    # handle('C://Users//dbettkk//Desktop//新建文件夹//resnet50_cudnn_time.txt')
    # handle('C://Users//dbettkk//Desktop//新建文件夹//resnet50_spmma_time.txt')
    # handle2('C://Users//dbettkk//Desktop//test_time//8.29//trans_time_16.txt')
    # handle_9_1('C://Users//dbettkk//Desktop//test_time//9.1//resnet50_cudnn_time.txt')
    handle3("data/trans_time.txt")
