
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>


struct Model {
    Model () {   
    seq = torch::nn::Sequential{};
        {% include 'workspace/models.cpp.tpl' %}
    }
    torch::Tensor forward(torch::Tensor x) {
        return torch::cat(x, dimension_)
    }

    torch::nn::Sequential seq = NULL;
};
