#ifndef _YOLOV5_CPP_DETECT_H
#define _YOLOV5_CPP_DETECT_H


#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

struct Detect : torch::nn::Module {
    Detect(int nc, int nl, float anchor[], int anchor_len,  bool inplace) {
        nc_ = nc;
        no_ = nc + 5;
        nl_ = nl;
        na_ = 3;  // 写死的
        for ( int i =0; i < nl; i++) {
            grid.push_back(torch::zeros(1));
        }
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto a = torch::from_blob(anchor, {anchor_len}, options).view({nl_, -1, 2});
        register_buffer("anchors", a);
        register_buffer("anchor_grid", a.clone().view({nl_, 1, -1, 1, 1, 2}));


        torch::nn::Conv2d conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 255, torch::ExpandingArray<2>({1, 1})).stride(torch::ExpandingArray<2>({1, 1})));
        torch::nn::Conv2d conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 255, torch::ExpandingArray<2>({1, 1})).stride(torch::ExpandingArray<2>({1, 1})));
        torch::nn::Conv2d conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 255, torch::ExpandingArray<2>({1, 1})).stride(torch::ExpandingArray<2>({1, 1})));
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        inplace_ = inplace;
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x[0] = conv1->forward(x[0]);
        auto sharp = x[0].sizes();
        x[0] = x[0].view({sharp[0], na_, no_, sharp[2], sharp[3]}).permute({0, 1, 3, 4, 2}).contiguous();

        x[1] = conv2->forward(x[1]);
        auto sharp2 = x[1].sizes();
        x[1] = x[1].view({sharp2[0], na_, no_, sharp2[2], sharp2[3]}).permute({0, 1, 3, 4, 2}).contiguous();

        x[2] = conv3->forward(x[2]);
        auto sharp3 = x[2].sizes();
        x[2] = x[2].view({sharp3[0], na_, no_, sharp3[2], sharp3[3]}).permute({0, 1, 3, 4, 2}).contiguous();

        return x;
    }

    int nc_;
    int no_;
    int nl_;
    int na_;
    std::vector<torch::Tensor> grid;
    torch::nn::ModuleList m1 = NULL;
    bool inplace_ = false;
    
    torch::nn::Conv2d conv1 = NULL;
    torch::nn::Conv2d conv2 = NULL;
    torch::nn::Conv2d conv3 = NULL;
};


#endif