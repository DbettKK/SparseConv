//
// Created by dbettkk on 2022/8/1.
//

#include "MyResNet.cuh"

void MyResNet::resnet50() {
    auto input = new MyTensor(1, 3, 224, 224, true);
    // 1. conv1
    auto conv1_out = new MyTensor(1, 64, 112, 112, true);
    input->conv2d(1, 64, 7, 7, 2, 3, conv1_out);
    // 2. maxpool
    auto maxpool_out = new MyTensor(1, 64, 56, 56, true);
    conv1_out->maxpool(3, 2, 1, maxpool_out);
    // 3. conv2_x 无残差连接





}

void MyResNet::bottleneck(MyTensor *input, int in_c, int out_c, int stride) {
    //self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1,
    //                               padding=0, bias=False)
    //        self.bn1 = nn.BatchNorm2d(out_channel)
    //        # ----
    //        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride,
    //                               padding=1, bias=False)
    //        self.bn2 = nn.BatchNorm2d(out_channel)
    //        # ----
    //        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1,
    //                               padding=1, bias=False)
    //        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
    //        # ----
    //        self.relu = nn.ReLU(inplace=True)
    //        self.downsample = downsample

    //input->conv2d(2, out_c, 1, 1, stride, 0, out);
}
