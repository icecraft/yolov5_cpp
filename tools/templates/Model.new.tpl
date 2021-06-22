
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

{% include 'layers.cpp.headers.tpl' %}

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


auto main() -> int {
    std::cout << "hello world!" << std::endl;
}