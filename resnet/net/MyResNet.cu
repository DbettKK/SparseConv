//
// Created by dbettkk on 2022/8/1.
//

#include "MyResNet.cuh"

void MyResNet::resnet50() {
    //CudaTime *times[6];
    //for (auto & time : times) time = new CudaTime();
    //times[0]->initAndStart();
    int batch = 4;
    auto input = new MyTensor(batch, 3, 224, 224, true, "data");
    // 1. conv1
    //times[1]->initAndStart();
    auto conv1_out = new MyTensor(batch, 64, 112, 112, true);
    input->conv2d(0, 64, 7, 7, 2, 3, conv1_out);
    //printf("conv1 time: %fms\n", times[1]->endAndGetTime());
    //conv1_out->print(true);
    // 2. maxpool
    auto maxpool_out = new MyTensor(batch, 64, 56, 56, true);
    conv1_out->maxpool(3, 2, 1, maxpool_out);
    conv1_out->free_tensor();
    // 3. conv2_x 无下采样   x3
    //times[2]->initAndStart();
    auto conv2_out1 = new MyTensor(batch, 256, 56, 56, true);
    auto conv2_out2 = new MyTensor(batch, 256, 56, 56, true);
    auto conv2_out3 = new MyTensor(batch, 256, 56, 56, true);
    bottleneck1(maxpool_out, conv2_out1, 0);
    bottleneck1(conv2_out1, conv2_out2, 1);
    bottleneck1(conv2_out2, conv2_out3, 2);
    maxpool_out->free_tensor();
    conv2_out1->free_tensor();
    conv2_out2->free_tensor();
    //printf("conv2 time: %fms\n", times[2]->endAndGetTime());
    // 4. conv3_x 有下采样   x4
    //times[3]->initAndStart();
    auto conv3_out1 = new MyTensor(batch, 512, 28, 28, true);
    auto conv3_out2 = new MyTensor(batch, 512, 28, 28, true);
    auto conv3_out3 = new MyTensor(batch, 512, 28, 28, true);
    auto conv3_out4 = new MyTensor(batch, 512, 28, 28, true);
    bottleneck2(conv2_out3, conv3_out1, 0);
    bottleneck2(conv3_out1, conv3_out2, 1);
    bottleneck2(conv3_out2, conv3_out3, 2);
    bottleneck2(conv3_out3, conv3_out4, 3);
    conv2_out3->free_tensor();
    conv3_out1->free_tensor();
    conv3_out2->free_tensor();
    conv3_out3->free_tensor();
    //printf("conv3 time: %fms\n", times[3]->endAndGetTime());
    // 5. conv4_x 有下采样  x6
    //times[4]->initAndStart();
    auto conv4_out1 = new MyTensor(batch, 1024, 14, 14, true);
    auto conv4_out2 = new MyTensor(batch, 1024, 14, 14, true);
    auto conv4_out3 = new MyTensor(batch, 1024, 14, 14, true);
    auto conv4_out4 = new MyTensor(batch, 1024, 14, 14, true);
    auto conv4_out5 = new MyTensor(batch, 1024, 14, 14, true);
    auto conv4_out6 = new MyTensor(batch, 1024, 14, 14, true);
    bottleneck3(conv3_out4, conv4_out1, 0);
    bottleneck3(conv4_out1, conv4_out2, 1);
    bottleneck3(conv4_out2, conv4_out3, 2);
    bottleneck3(conv4_out3, conv4_out4, 3);
    bottleneck3(conv4_out4, conv4_out5, 4);
    bottleneck3(conv4_out5, conv4_out6, 5);
    conv3_out4->free_tensor();
    conv4_out1->free_tensor();
    conv4_out2->free_tensor();
    conv4_out3->free_tensor();
    conv4_out4->free_tensor();
    conv4_out5->free_tensor();
    //printf("conv4 time: %fms\n", times[4]->endAndGetTime());
    // 6. conv5_x 有下采样  x3
    //times[5]->initAndStart();
    auto conv5_out1 = new MyTensor(batch, 2048, 7, 7, true);
    auto conv5_out2 = new MyTensor(batch, 2048, 7, 7, true);
    auto conv5_out3 = new MyTensor(batch, 2048, 7, 7, true);
    bottleneck4(conv4_out6, conv5_out1, 0);
    bottleneck4(conv5_out1, conv5_out2, 1);
    bottleneck4(conv5_out2, conv5_out3, 2);
    conv4_out6->free_tensor();
    conv5_out1->free_tensor();
    conv5_out2->free_tensor();
    //printf("conv5 time: %fms\n", times[5]->endAndGetTime());
    // avgpool
    auto avg_out = new MyTensor(batch, 2048, 1, 1, true);
    conv5_out3->avgpool(avg_out);
    conv5_out3->free_tensor();
    // liner 2048, 1000
    auto liner_out = new MyTensor(batch, 1000, 1, 1, true);
    cublas_gemm_device(avg_out->getTensor(), W_fc->getTensor(), 1, 2048, 1000, liner_out->getTensor());
    avg_out->free_tensor();
    // softmax
    auto softmax_out = new MyTensor(batch, 1000, 1, 1, true);
    softmax_cudnn(liner_out->getTensor(), batch, 1000, 1, 1, softmax_out->getTensor());
    liner_out->free_tensor();


    // softmax_out->print(true);
}

void MyResNet::bottleneck1(MyTensor *input, MyTensor *output, int times) {
    int batch = input->getBatch();
    auto conv1_out = new MyTensor(batch, 64, 56, 56, true);
    input->conv2d(1 + 3 * times, 64, 1, 1, 1, 0, conv1_out);
    auto bn1_out = new MyTensor(batch, 64, 56, 56, true);
    conv1_out->batchNorm(64, bn1_out);
    auto relu1_out = new MyTensor(batch, 64, 56, 56, true);
    bn1_out->relu(relu1_out);

    auto conv2_out = new MyTensor(batch, 64, 56, 56, true);
    relu1_out->conv2d(2 + 3 * times, 64, 3, 3, 1, 1, conv2_out);
    auto bn2_out = new MyTensor(batch, 64, 56, 56, true);
    conv2_out->batchNorm(64, bn2_out);
    auto relu2_out = new MyTensor(batch, 64, 56, 56, true);
    bn2_out->relu(relu2_out);

    auto conv3_out = new MyTensor(batch, 256, 56, 56, true);
    relu2_out->conv2d(3 + 3 * times, 64, 1, 1, 1, 0, conv3_out);
    auto bn3_out = new MyTensor(batch, 256, 56, 56, true);
    conv3_out->batchNorm(64, bn3_out);

    // input 下采样
    auto downsample = new MyTensor(batch, 256, 56, 56, true);
    input->conv2d(100 + times, 256, 1, 1, 1, 0, downsample);
    auto bn_down = new MyTensor(batch, 256, 56, 56, true);
    downsample->batchNorm(256, bn_down);

    auto add_out = new MyTensor(batch, 256, 56, 56, true);
    bn3_out->addTensor(bn_down, add_out);

    auto relu3_out = new MyTensor(batch, 256, 56, 56, true);
    add_out->relu(relu3_out);

    relu3_out->copy(output);

    // free
    conv1_out->free_tensor();
    conv2_out->free_tensor();
    conv3_out->free_tensor();
    bn1_out->free_tensor();
    bn2_out->free_tensor();
    bn3_out->free_tensor();
    relu1_out->free_tensor();
    relu2_out->free_tensor();
    relu3_out->free_tensor();
    add_out->free_tensor();
    bn_down->free_tensor();
    downsample->free_tensor();
}

void MyResNet::bottleneck2(MyTensor *input, MyTensor *output, int times) {
    int batch = input->getBatch();
    auto conv1_out = new MyTensor(batch, 128, 56, 56, true);
    input->conv2d(10 + 3 * times, 128, 1, 1, 1, 0, conv1_out);
    auto bn1_out = new MyTensor(batch, 128, 56, 56, true);
    conv1_out->batchNorm(128, bn1_out);
    auto relu1_out = new MyTensor(batch, 128, 56, 56, true);
    bn1_out->relu(relu1_out);

    auto conv2_out = new MyTensor(batch, 128, 28, 28, true);
    relu1_out->conv2d(11 + 3 * times, 128, 3, 3, 2, 1, conv2_out);
    auto bn2_out = new MyTensor(batch, 128, 28, 28, true);
    conv2_out->batchNorm(128, bn2_out);
    auto relu2_out = new MyTensor(batch, 128, 28, 28, true);
    bn2_out->relu(relu2_out);

    auto conv3_out = new MyTensor(batch, 512, 28, 28, true);
    relu2_out->conv2d(12 + 3 * times, 512, 1, 1, 1, 0, conv3_out);
    auto bn3_out = new MyTensor(batch, 512, 28, 28, true);
    conv3_out->batchNorm(512, bn3_out);

    // input 下采样
    auto downsample = new MyTensor(batch, 512, 28, 28, true);
    input->conv2d(103 + times, 512, 1, 1, 2, 0, downsample);
    auto bn_down = new MyTensor(batch, 512, 28, 28, true);
    downsample->batchNorm(512, bn_down);

    auto add_out = new MyTensor(batch, 512, 28, 28, true);
    bn3_out->addTensor(bn_down, add_out);

    auto relu3_out = new MyTensor(batch, 512, 28, 28, true);
    add_out->relu(relu3_out);

    relu3_out->copy(output);

    // free
    conv1_out->free_tensor();
    conv2_out->free_tensor();
    conv3_out->free_tensor();
    bn1_out->free_tensor();
    bn2_out->free_tensor();
    bn3_out->free_tensor();
    relu1_out->free_tensor();
    relu2_out->free_tensor();
    relu3_out->free_tensor();
    add_out->free_tensor();
    downsample->free_tensor();
    bn_down->free_tensor();
}

void MyResNet::bottleneck3(MyTensor *input, MyTensor *output, int times) {
    int batch = input->getBatch();
    auto conv1_out = new MyTensor(batch, 256, 28, 28, true);
    input->conv2d(22 + 3 * times, 256, 1, 1, 1, 0, conv1_out);
    auto bn1_out = new MyTensor(batch, 256, 28, 28, true);
    conv1_out->batchNorm(256, bn1_out);
    auto relu1_out = new MyTensor(batch, 256, 28, 28, true);
    bn1_out->relu(relu1_out);

    auto conv2_out = new MyTensor(batch, 256, 14, 14, true);
    relu1_out->conv2d(23 + 3 * times, 256, 3, 3, 2, 1, conv2_out);
    auto bn2_out = new MyTensor(batch, 256, 14, 14, true);
    conv2_out->batchNorm(256, bn2_out);
    auto relu2_out = new MyTensor(batch, 256, 14, 14, true);
    bn2_out->relu(relu2_out);

    auto conv3_out = new MyTensor(batch, 1024, 14, 14, true);
    relu2_out->conv2d(24 + 3 * times, 1024, 1, 1, 1, 0, conv3_out);
    auto bn3_out = new MyTensor(batch, 1024, 14, 14, true);
    conv3_out->batchNorm(1024, bn3_out);

    // input 下采样
    auto downsample = new MyTensor(batch, 1024, 14, 14, true);
    input->conv2d(107 + times, 1024, 1, 1, 2, 0, downsample);
    auto bn_down = new MyTensor(batch, 1024, 14, 14, true);
    downsample->batchNorm(1024, bn_down);

    auto add_out = new MyTensor(batch, 1024, 14, 14, true);
    bn3_out->addTensor(bn_down, add_out);

    auto relu3_out = new MyTensor(batch, 1024, 14, 14, true);
    add_out->relu(relu3_out);

    relu3_out->copy(output);

    // free
    conv1_out->free_tensor();
    conv2_out->free_tensor();
    conv3_out->free_tensor();
    bn1_out->free_tensor();
    bn2_out->free_tensor();
    bn3_out->free_tensor();
    relu1_out->free_tensor();
    relu2_out->free_tensor();
    relu3_out->free_tensor();
    add_out->free_tensor();
    downsample->free_tensor();
    bn_down->free_tensor();
}

void MyResNet::bottleneck4(MyTensor *input, MyTensor *output, int times) {
    int batch = input->getBatch();
    auto conv1_out = new MyTensor(batch, 512, 14, 14, true);
    input->conv2d(40 + 3 * times, 512, 1, 1, 1, 0, conv1_out);
    auto bn1_out = new MyTensor(batch, 512, 14, 14, true);
    conv1_out->batchNorm(512, bn1_out);
    auto relu1_out = new MyTensor(batch, 512, 14, 14, true);
    bn1_out->relu(relu1_out);

    auto conv2_out = new MyTensor(batch, 512, 7, 7, true);
    relu1_out->conv2d(41 + 3 * times, 512, 3, 3, 2, 1, conv2_out);
    auto bn2_out = new MyTensor(batch, 512, 7, 7, true);
    conv2_out->batchNorm(512, bn2_out);
    auto relu2_out = new MyTensor(batch, 512, 7, 7, true);
    bn2_out->relu(relu2_out);

    auto conv3_out = new MyTensor(batch, 2048, 7, 7, true);
    relu2_out->conv2d(42 + 3 * times, 2048, 1, 1, 1, 0, conv3_out);
    auto bn3_out = new MyTensor(batch, 2048, 7, 7, true);
    conv3_out->batchNorm(2048, bn3_out);

    // input 下采样
    auto downsample = new MyTensor(batch, 2048, 7, 7, true);
    input->conv2d(113 + times, 2048, 1, 1, 2, 0, downsample);
    auto bn_down = new MyTensor(batch, 2048, 7, 7, true);
    downsample->batchNorm(2048, bn_down);

    auto add_out = new MyTensor(batch, 2048, 7, 7, true);
    bn3_out->addTensor(bn_down, add_out);

    auto relu3_out = new MyTensor(batch, 2048, 7, 7, true);
    add_out->relu(relu3_out);

    relu3_out->copy(output);

    // free
    conv1_out->free_tensor();
    conv2_out->free_tensor();
    conv3_out->free_tensor();
    bn1_out->free_tensor();
    bn2_out->free_tensor();
    bn3_out->free_tensor();
    relu1_out->free_tensor();
    relu2_out->free_tensor();
    relu3_out->free_tensor();
    add_out->free_tensor();
    downsample->free_tensor();
    bn_down->free_tensor();
}

