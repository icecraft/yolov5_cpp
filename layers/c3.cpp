
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

struct C3 : torch::nn::Module {
    C3 (int64_t input_channels, int64_t output_channels, int n, bool shortcut, int g, float eps) {
        int64_t hidden = int(output_channels * eps);

        conv1 = torch::nn:Conv2d(torch::nn::Conv2dOptions(input_channels, hidden, (1, 1)).
                                stride((1, 1)).
                                bias(false));

        conv2 = torch::nn:Conv2d(torch::nn::Conv2dOptions(input_channels, hidden, (1, 1)).
                        stride((1, 1)).
                        bias(false));
                   
        conv3 = torch::nn:Conv2d(torch::nn::Conv2dOptions(2 * hidden, output_channels, (1, 1)).
                        stride((1, 1)).
                        bias(false));

        seq1 = torch::nn::Sequential{}

        for (int i = 0 ;i < n; i++) {
            bottleneck = Bottleneck(hidden, hidden, shortcut, g, eps);
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