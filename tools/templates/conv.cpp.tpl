
{% from "_standard.macro.tpl" import conv2d with context %}

#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>


struct Conv : torch::nn::Module {
  Conv()
      : conv1({{ conv2d(in_channels, out_channels, kernel_size, padding, bias, groups, stride)}}),
        bn1(torch:nn:BatchNorm2d({{out_channels}})),
        silu1(torch::nn::SiLu()),
        id1(torch::nn::Identity()) {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("silu1", silu1);
    register_module("id1", id1);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = silu1->forward(x);
    return x
  }

  torch::Tensor fuseforward(torch::Tensor x) {
    x = conv1->forward(x);
    x = silu1->forward(x);
    return x
  }

  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm2d bn1
  torch::nn::SiLU silu1
  torch::nn::Identity id1;
};


