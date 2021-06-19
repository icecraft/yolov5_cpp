
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>


struct C3 : torch::nn::Module {
    C3 (int64_t input_channels, int64_t output_channels, 
        torch::ExpandingArray<2> kernel_size, 
        torch::ExpandingArray<2> stride,
        torch::ExpandingArray<2> padding,
        torch::ExpandingArray<2> dilation,
        int64_t groups,
        bool bias,
        int n, bool shortcut, int64_t groups2, float eps) {
        int64_t hidden = int(output_channels * eps);

        conv1 = &Conv(input_channels, hidden, kernel_size, stride, padding, dilation, groups, bias);
        conv2 = &Conv(input_channels, hidden, kernel_size, stride, padding, dilation, groups, bias);
        conv3 = &Conv(2 * hidden, output_channels, kernel_size, stride, padding, dilation, groups, bias);

        seq1 = torch::nn::Sequential{}

        for (int i = 0 ;i < n; i++) {
            bottleneck = Bottleneck(hidden, hidden, shortcut, groups2, eps);
            seq1.push_back(bottleneck);
        }

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("seq1", seq1);
    }

    torch::Tensor forward(torch::Tensor x) {
        //       return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        torch::Tensor x1 = seql->forward(conv1->forward(x));
        torch::Tensor x2 = conv2->forward(x);
        torch::Tensor x3 = torch::cat({x1, x2} , 1)
        x = conv3->forward(x);
        return x;
    }

    Conv conv1 = NULL;
    Conv conv2 = NULL;
    Conv conv3 = NULL;
    torch::nn::Sequential seq1 = NULL;
};