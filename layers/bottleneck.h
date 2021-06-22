#ifndef _YOLOV5_CPP_BOTTLENECK_H
#define _YOLOV5_CPP_BOTTLENECK_H


#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>


struct Bottleneck : torch::nn::Module {
    Bottleneck (int64_t input_channels, int64_t output_channels, bool shortcut, int64_t g, float eps) {
        int64_t hidden = int(output_channels * eps);

        conv1 = &Conv(input_channels, hidden, 
                    torch::ExpandingArray<2>({1, 1}),
                    torch::ExpandingArray<2>({1, 1}),
                    torch::ExpandingArray<2>({0, 0}),
                    torch::ExpandingArray<2>({1, 1}),
                    1, 
                    false
                    );
        conv2 = &Conv(input_channels, hidden, 
                    torch::ExpandingArray<2>({3, 3}),
                    torch::ExpandingArray<2>({1, 1}),
                    torch::ExpandingArray<2>({1, 1}),
                    torch::ExpandingArray<2>({1, 1}),
                    g, 
                    false
                    );

        if (shortcut && input_channels == output_channels) {
            pAdd = true;
        }

        register_module("conv1", conv1);
        register_module("conv2", conv2);
   
    }

    // 假定只有 3 个，必要时手动修改吧
    torch::Tensor forward(torch::Tensor x) {
        if pAdd {
            torch::Tensor x_ = torch::clone(x);
            x = conv1->forward(x);
            x = conv2->forward(x);
            return x_ + x;
        } else {
            x = conv1->forward(x);
            x = conv2->forward(x);
            return x;
        }
    }

    Conv conv1 = NULL;
    Conv conv2 = NULL;
    bool pAdd = false;
};

#endif