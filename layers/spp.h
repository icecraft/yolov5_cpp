#ifndef _YOLOV5_CPP_SPP_H
#define _YOLOV5_CPP_SPP_H


#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

struct SPP : torch::nn::Module {
    SPP (int64_t input_channels, int64_t output_channels,
            torch::ExpandingArray<2> kernel_size, 
            torch::ExpandingArray<2> stride,
            torch::ExpandingArray<2> padding,
            torch::ExpandingArray<2> dilation,
            int64_t groups,
            bool bias,
            std::vector<float> pool_kernel_size) 
     {
        int64_t hidden = input_channels / 2;

        conv1 = std::make_shared<Conv>(input_channels, hidden, kernel_size, stride, padding, dilation, groups, bias);
        conv2 = std::make_shared<Conv>((pool_kernel_size.size()+1)*hidden, output_channels, kernel_size, stride, padding, dilation, groups, bias);

        m1 = torch::nn::ModuleList();

        for (auto x: pool_kernel_size) {
            pool = torch::nn::MaxPool2dOptions(x).stride(1).padding(x/2);
            m1->push_back(pool);
        }

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("m1", m1);
    }

    //TODO: 假定只有 3 个，必要时手动修改吧。 more elegant
    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);
        std::vector<torch::Tensor> arr;

        for (auto proc : m1) {
            arr.push_back(proc->forward(x));
        }
        x = torch::cat({x, arr[0], arr[1], arr[2]}, 1);
        x = conv2->forward(x);
        return x;
    }

    std::shared_ptr<Conv> conv1 = NULL;
    std::shared_ptr<Conv> conv2 = NULL;
    torch::nn::ModuleList m1 = NULL;
};

#endif