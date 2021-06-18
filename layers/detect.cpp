
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

struct Detect : torch::nn::Module {
    Detect(int nc, int nl, int[] anchor, int anchor_len,  bool inplace) {
        nc_ = nc;
        no_ = nc + 5;
        nl_ = nl;
        na_ = 3;  // 写死的
        for ( int i =0; i < nl; i++) {
            grid.push_back(torch::zeors(1));
        }
        auto a = torch::from_blob(anchor,{anchor_len},torch::kFloat32).view({nl_, -1, 2});
        register_buffer("anchors", a);
        register_buffer("anchor_grid", a.clone().view({nl_, 1, -1, 1, 1, 2});

        m1 = torch::nn::ModuleList();
        torch::nn::Conv2d conv1 = torch::nn:Conv2d(torch::nn::Conv2dOptions(128, 255, 1, 1).stride(1, 1);
        torch::nn::Conv2d conv2 = torch::nn:Conv2d(torch::nn::Conv2dOptions(256, 255, 1, 1).stride(1, 1);
        torch::nn::Conv2d conv3 = torch::nn:Conv2d(torch::nn::Conv2dOptions(512, 255, 1, 1).stride(1, 1);
        m1.push_back(conv1);
        m1.push_back(conv2);
        m1.push_back(conv3);
        register_module("m1", m1);
        inplace_ = inplace;
    }
    
    torch::Tensor forward(torch::Tensor x) {
        int i = 0;
        for (auto proc : m1) {
            x[i] = proc->forward(x[i]);
            auto sharp = x[i].shape();
            x[i] = x[i].view(sharp[0], na_, no_, sharp[2], sharp[3]).permute({0, 1, 3, 4, 2}).contiguous();
            i ++;
        }
       return x;
    }

    int nc_;
    int no_;
    int nl_;
    int na_;
    std::Vector<torch::Tensor> grid;
    torch::nn::ModuleList m1 = NULL;
    bool inplace_ = false;
};


