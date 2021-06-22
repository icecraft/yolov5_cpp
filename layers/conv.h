#ifndef _YOLOV5_CPP_CONV_H
#define _YOLOV5_CPP_CONV_H

#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

struct Conv : torch::nn::Module {
    Conv(int64_t input_channels, int64_t output_channels,
            torch::ExpandingArray<2> kernel_size, 
            torch::ExpandingArray<2> stride,
            torch::ExpandingArray<2> padding,
            torch::ExpandingArray<2> dilation,
            int64_t groups,
            bool bias) {
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size).
                                stride(stride).
                                padding(padding).
                                dilation(dilation).
                                groups(groups).
                                bias(bias));
        bn1 = torch::nn::BatchNorm2d(output_channels);
        silu1 = torch::nn::SiLU();
        id1 = torch::nn::Identity();

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("silu1", silu1);
        register_module("id1", id1);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = silu1->forward(x);
        return x;
    }

    torch::Tensor fuseforward(torch::Tensor x) {
        x = conv1->forward(x);
        x = silu1->forward(x);
        return x;
    }

    torch::nn::Conv2d conv1 = NULL;
    torch::nn::BatchNorm2d bn1 = NULL;
    torch::nn::SiLU silu1 = NULL;
    torch::nn::Identity id1 = NULL;
};

#endif