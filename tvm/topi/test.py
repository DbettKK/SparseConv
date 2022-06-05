from conv2d import topi_cuda, time


def tvm_test_vgg():
    vgg_path = '../data/vgg'
    topi_cuda([1, 3, 224, 224], [64, 3, 3, 3], 1, 1, vgg_path + '/data1.bin', vgg_path + '/filter1.bin')
    topi_cuda([1, 64, 224, 224], [64, 64, 3, 3], 1, 1, vgg_path + '/data2.bin', vgg_path + '/filter2.bin')
    topi_cuda([1, 64, 112, 112], [128, 64, 3, 3], 1, 1, vgg_path + '/data3.bin', vgg_path + '/filter3.bin')
    topi_cuda([1, 128, 112, 112], [128, 128, 3, 3], 1, 1, vgg_path + '/data4.bin', vgg_path + '/filter4.bin')
    topi_cuda([1, 128, 56, 56], [256, 128, 3, 3], 1, 1, vgg_path + '/data5.bin', vgg_path + '/filter5.bin')
    topi_cuda([1, 256, 56, 56], [256, 256, 3, 3], 1, 1, vgg_path + '/data6.bin', vgg_path + '/filter6.bin')
    topi_cuda([1, 256, 56, 56], [256, 256, 3, 3], 1, 1, vgg_path + '/data7.bin', vgg_path + '/filter7.bin')
    topi_cuda([1, 256, 28, 28], [512, 256, 3, 3], 1, 1, vgg_path + '/data8.bin', vgg_path + '/filter8.bin')
    topi_cuda([1, 512, 28, 28], [512, 512, 3, 3], 1, 1, vgg_path + '/data9.bin', vgg_path + '/filter9.bin')
    topi_cuda([1, 512, 28, 28], [512, 512, 3, 3], 1, 1, vgg_path + '/data10.bin', vgg_path + '/filter10.bin')
    topi_cuda([1, 512, 14, 14], [512, 512, 3, 3], 1, 1, vgg_path + '/data11.bin', vgg_path + '/filter11.bin')
    topi_cuda([1, 512, 14, 14], [512, 512, 3, 3], 1, 1, vgg_path + '/data12.bin', vgg_path + '/filter12.bin')
    topi_cuda([1, 512, 14, 14], [512, 512, 3, 3], 1, 1, vgg_path + '/data13.bin', vgg_path + '/filter13.bin')


def tvm_test_resnet18():
    resnet_path = "../data/resnet"
    topi_cuda([1, 3, 224, 224], [64, 3, 7, 7], 3, 2, resnet_path + "/data1.bin", resnet_path + "/filter1.bin")
    topi_cuda([1, 64, 56, 56], [64, 64, 3, 3], 1, 1, resnet_path + "/data2.bin", resnet_path + "/filter2.bin")
    topi_cuda([1, 64, 56, 56], [64, 64, 3, 3], 1, 1, resnet_path + "/data3.bin", resnet_path + "/filter3.bin")
    topi_cuda([1, 64, 56, 56], [64, 64, 3, 3], 1, 1, resnet_path + "/data4.bin", resnet_path + "/filter4.bin")
    topi_cuda([1, 64, 56, 56], [64, 64, 3, 3], 1, 1, resnet_path + "/data5.bin", resnet_path + "/filter5.bin")
    topi_cuda([1, 64, 56, 56], [128, 64, 3, 3], 1, 2, resnet_path + "/data6.bin", resnet_path + "/filter6.bin")
    topi_cuda([1, 128, 28, 28], [128, 128, 3, 3], 1, 1, resnet_path + "/data7.bin", resnet_path + "/filter7.bin")
    topi_cuda([1, 128, 28, 28], [128, 128, 3, 3], 1, 1, resnet_path + "/data8.bin", resnet_path + "/filter8.bin")
    topi_cuda([1, 128, 28, 28], [128, 128, 3, 3], 1, 1, resnet_path + "/data9.bin", resnet_path + "/filter9.bin")
    topi_cuda([1, 128, 28, 28], [256, 128, 3, 3], 1, 2, resnet_path + "/data10.bin", resnet_path + "/filter10.bin")
    topi_cuda([1, 256, 14, 14], [256, 256, 3, 3], 1, 1, resnet_path + "/data11.bin", resnet_path + "/filter11.bin")
    topi_cuda([1, 256, 14, 14], [256, 256, 3, 3], 1, 1, resnet_path + "/data12.bin", resnet_path + "/filter12.bin")
    topi_cuda([1, 256, 14, 14], [256, 256, 3, 3], 1, 1, resnet_path + "/data13.bin", resnet_path + "/filter13.bin")
    topi_cuda([1, 256, 14, 14], [512, 256, 3, 3], 1, 2, resnet_path + "/data14.bin", resnet_path + "/filter14.bin")
    topi_cuda([1, 512, 7, 7], [512, 512, 3, 3], 1, 1, resnet_path + "/data15.bin", resnet_path + "/filter15.bin")
    topi_cuda([1, 512, 7, 7], [512, 512, 3, 3], 1, 1, resnet_path + "/data16.bin", resnet_path + "/filter16.bin")
    topi_cuda([1, 512, 7, 7], [512, 512, 3, 3], 1, 1, resnet_path + "/data17.bin", resnet_path + "/filter17.bin")


def tvm_test_alexnet():
    alexnet_path = "../data/alex"
    topi_cuda([1, 3, 227, 227], [96, 3, 11, 11], 0, 4, alexnet_path + "/data1.bin", alexnet_path + "/filter1.bin")
    topi_cuda([1, 96, 27, 27], [256, 96, 5, 5], 2, 1, alexnet_path + "/data2.bin", alexnet_path + "/filter2.bin")
    topi_cuda([1, 256, 13, 13], [384, 256, 3, 3], 1, 1, alexnet_path + "/data3.bin", alexnet_path + "/filter3.bin")
    topi_cuda([1, 384, 13, 13], [384, 384, 3, 3], 1, 1, alexnet_path + "/data4.bin", alexnet_path + "/filter4.bin")
    topi_cuda([1, 384, 13, 13], [256, 384, 3, 3], 1, 1, alexnet_path + "/data5.bin", alexnet_path + "/filter5.bin")


if __name__ == '__main__':
    tvm_test_vgg()
    print(time)