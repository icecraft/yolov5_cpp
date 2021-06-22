#ifndef _YOLOV5_CPP_CONCAT_H
#define _YOLOV5_CPP_CONCAT_H


#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>


struct Concat : torch::nn::Module {
    Concat(int dimension) {
        dimension_ = dimension;
    }

    torch::Tensor forward(torch::Tensor x) {
        return torch::cat(x, dimension_)
    }

    int dimension_;
};

#endif