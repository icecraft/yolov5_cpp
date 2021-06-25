
#ifndef _YOLOV5_CPP_C3_H
#define _YOLOV5_CPP_C3_H

#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

# include "conv.h"
# include "bottleneck.h"

struct C3 : torch::nn::Module {
    C3 (int64_t input_channels, int64_t output_channels, 
        torch::ExpandingArray<2> kernel_size, 
        torch::ExpandingArray<2> stride,
        torch::ExpandingArray<2> padding,
        torch::ExpandingArray<2> dilation,
        int64_t groups,
        bool bias,
        int n, bool shortcut, int64_t groups2, float eps) {
        int64_t hidden = int64_t(output_channels * eps);

        conv1 = std::make_shared<Conv>(input_channels, hidden, kernel_size, stride, padding, dilation, groups, bias);
        conv2 = std::make_shared<Conv>(input_channels, hidden, kernel_size, stride, padding, dilation, groups, bias);
        conv3 = std::make_shared<Conv>(2 * hidden, output_channels, kernel_size, stride, padding, dilation, groups, bias);

        seq1 = torch::nn::Sequential{};

        for (int i = 0 ;i < n; i++) {
            Bottleneck bottleneck = Bottleneck(hidden, hidden, shortcut, groups2, eps);
            seq1->push_back(bottleneck);
        }

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("seq1", seq1);
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor x1 = seq1->forward(conv1->forward(x));
        torch::Tensor x2 = conv2->forward(x);
        torch::Tensor x3 = torch::cat({x1, x2} , 1);
        x = conv3->forward(x3);
        return x;
    }

    std::shared_ptr<Conv> conv1 = NULL;
    std::shared_ptr<Conv> conv2 = NULL;
    std::shared_ptr<Conv> conv3 = NULL;
    torch::nn::Sequential seq1 = NULL;
};

#endif