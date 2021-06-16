


#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>


/*

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))

*/

in_channels, out_channels, kernel_size, padding, bias, groups, stride

struct Focus:torch::nn::Module {
  Focus()
      : conv1(Conv(in_channels * 4, out_channels, kernel_size, padding, bias=False )),
    register_module("conv1", conv1);

  torch::Tensor forward(torch::Tensor x) {
    x = conv1->forward(x);
    return x;
  }

  Conv conv1;
};

