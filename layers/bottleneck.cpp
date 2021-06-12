
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

struct Bottleneck : torch::nn::Module {
    Bottleneck (int64_t input_channels, int64_t output_channels, bool shortcut, int g, float eps) {
        int64_t hidden = int(output_channels * e);

        conv1 = torch::nn:Conv2d(torch::nn::Conv2dOptions(input_channels, hidden, (1, 1)).
                                stride((1, 1)).
                                bias(false));
        conv2 = torch::nn:Conv2d(torch::nn::Conv2dOptions(hidden, output_channels, (1, 1)).
                                stride((1, 1)).
                                bias(false));
    
        if (hidden == output_channels) && (hidden == input_channels) && shortcut {
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