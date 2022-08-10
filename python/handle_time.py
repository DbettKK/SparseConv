import re


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


if __name__ == '__main__':
    # handle('C://Users//dbettkk//Desktop//新建文件夹//resnet50_cudnn_time.txt')
    handle('C://Users//dbettkk//Desktop//新建文件夹//resnet50_spmma_time.txt')
