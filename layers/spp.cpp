
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

struct SPP : torch::nn::Module {
    SPP (int64_t input_channels, int64_t output_channels, int64_t kernels[] ,int kernel_nums) {
        int64_t hidden = input_channels / 2;

        conv1 = torch::nn:Conv2d(torch::nn::Conv2dOptions(input_channels, hidden, (1, 1)).
                                stride((1, 1)).
                                bias(false));
        conv2 = torch::nn:Conv2d(torch::nn::Conv2dOptions(hidden, output_channels, (1, 1)).
                                stride((1, 1)).
                                bias(false));
        m1 = torch::nn::ModuleList();

        for (int i = 0 ;i < kernel_size; i++) {
            int x = kernels[i];
            pool = torch::nn::MaxPool2dOptions(x).stride(1).padding(x/2);
            m1.push_back(pool);
        }

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("m1", m1);
    }

    // 假定只有 3 个，必要时手动修改吧
    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);
        x = torch::cat({x, m[1]->forward(x), m[2]->forward(x), m[3]->forward(x)}, 1)
        x = conv2->forward(x);
        return x;
    }

    Conv conv1 = NULL;
    Conv conv2 = NULL;
    torch::nn::ModuleList m1 = NULL;
};