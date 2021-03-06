#ifndef _YOLOV5_CPP_FOCUS_H
#define _YOLOV5_CPP_FOCUS_H

#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace torch::indexing;

struct Focus : torch::nn::Module {
    Focus (int64_t input_channels, int64_t output_channels,
            torch::ExpandingArray<2> kernel_size, 
            torch::ExpandingArray<2> stride,
            torch::ExpandingArray<2> padding,
            torch::ExpandingArray<2> dilation,
            int64_t groups,
            bool bias) {
        conv1 = std::make_shared<Conv>(input_channels, output_channels, kernel_size, stride, padding, dilation, groups, bias);
        register_module("conv1", conv1);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(torch::cat({x.index({"...", Slice(None, None, 2), Slice(None, None, 2)}),
                                        x.index({"...", Slice(1, None, 2), Slice(None, None, 2)}),
                                        x.index({"...", Slice(None, None, 2), Slice(1, None, 2)}),
                                        x.index({"...", Slice(1, None, 2), Slice(1, None, 2)})}, 1));
        return x;
    }

    std::shared_ptr<Conv> conv1 = NULL;
};

#endif