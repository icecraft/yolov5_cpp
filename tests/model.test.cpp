
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>


# include "../layers/bottleneck.h"
# include "../layers/c3.h"
# include "../layers/concat.h"
# include "../layers/conv.h"
# include "../layers/detect.h"
# include "../layers/focus.h"
# include "../layers/spp.h"

struct Model {
    Model () {   
    seq = torch::nn::Sequential{};
        
        std::shared_ptr<Focus> focus_1 = std::make_shared<Focus>(12, 32, 
                              torch::ExpandingArray<2>({3, 3}),
                              torch::ExpandingArray<2>({1, 1}),
                              torch::ExpandingArray<2>({1, 1}),
                              torch::ExpandingArray<2>({1, 1}),
                              1,
                              false
                            );
                            
        seq->push_back(focus_1);
        std::shared_ptr<Conv> conv_1 = std::make_shared<Conv>(32, 64, 
                          torch::ExpandingArray<2>({3, 3}),
                          torch::ExpandingArray<2>({2, 2}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq->push_back(conv_1);
        std::shared_ptr<C3> c3_1 = std::make_shared<C3>(64, 32, 
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({0, 0}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false,
                          1,
                          true,
                          1,
                          0.5
                            );
                            
        seq->push_back(c3_1);
        std::shared_ptr<Conv> conv_2 = std::make_shared<Conv>(64, 128, 
                          torch::ExpandingArray<2>({3, 3}),
                          torch::ExpandingArray<2>({2, 2}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq->push_back(conv_2);
        std::shared_ptr<C3> c3_2 = std::make_shared<C3>(128, 64, 
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({0, 0}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false,
                          3,
                          true,
                          1,
                          0.5
                            );
                            
        seq->push_back(c3_2);
        std::shared_ptr<Conv> conv_3 = std::make_shared<Conv>(128, 256, 
                          torch::ExpandingArray<2>({3, 3}),
                          torch::ExpandingArray<2>({2, 2}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq->push_back(conv_3);
        std::shared_ptr<C3> c3_3 = std::make_shared<C3>(256, 128, 
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({0, 0}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false,
                          3,
                          true,
                          1,
                          0.5
                            );
                            
        seq->push_back(c3_3);
        std::shared_ptr<Conv> conv_4 = std::make_shared<Conv>(256, 512, 
                          torch::ExpandingArray<2>({3, 3}),
                          torch::ExpandingArray<2>({2, 2}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq->push_back(conv_4);
        std::shared_ptr<SPP> spp_1 = std::make_shared<SPP>(512, 256,
                    torch::ExpandingArray<2>({1, 1}),
                    torch::ExpandingArray<2>({1, 1}),
                    torch::ExpandingArray<2>({0, 0}),
                    torch::ExpandingArray<2>({1, 1}),
                    1,
                    false,
                    std::vector<float>({5, 9, 13})
                    );

        seq->push_back(spp_1);
        std::shared_ptr<C3> c3_4 = std::make_shared<C3>(512, 256, 
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({0, 0}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false,
                          1,
                          false,
                          1,
                          0.5
                            );
                            
        seq->push_back(c3_4);
        std::shared_ptr<Conv> conv_5 = std::make_shared<Conv>(512, 256, 
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({0, 0}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq->push_back(conv_5);
        std::shared_ptr<Concat> concat_1 = std::make_shared<Concat>(1);

        seq->push_back(concat_1);
        std::shared_ptr<C3> c3_5 = std::make_shared<C3>(512, 128, 
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({0, 0}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false,
                          1,
                          false,
                          1,
                          0.25
                            );
                            
        seq->push_back(c3_5);
        std::shared_ptr<Conv> conv_6 = std::make_shared<Conv>(256, 128, 
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({0, 0}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq->push_back(conv_6);
        std::shared_ptr<Concat> concat_2 = std::make_shared<Concat>(1);

        seq->push_back(concat_2);
        std::shared_ptr<C3> c3_6 = std::make_shared<C3>(256, 64, 
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({0, 0}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false,
                          1,
                          false,
                          1,
                          0.25
                            );
                            
        seq->push_back(c3_6);
        std::shared_ptr<Conv> conv_7 = std::make_shared<Conv>(128, 128, 
                          torch::ExpandingArray<2>({3, 3}),
                          torch::ExpandingArray<2>({2, 2}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq->push_back(conv_7);
        std::shared_ptr<Concat> concat_3 = std::make_shared<Concat>(1);

        seq->push_back(concat_3);
        std::shared_ptr<C3> c3_7 = std::make_shared<C3>(256, 128, 
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({0, 0}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false,
                          1,
                          false,
                          1,
                          0.5
                            );
                            
        seq->push_back(c3_7);
        std::shared_ptr<Conv> conv_8 = std::make_shared<Conv>(256, 256, 
                          torch::ExpandingArray<2>({3, 3}),
                          torch::ExpandingArray<2>({2, 2}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq->push_back(conv_8);
        std::shared_ptr<Concat> concat_4 = std::make_shared<Concat>(1);

        seq->push_back(concat_4);
        std::shared_ptr<C3> c3_8 = std::make_shared<C3>(512, 256, 
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({0, 0}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false,
                          1,
                          false,
                          1,
                          0.5
                            );
                            
        seq->push_back(c3_8);
        std::shared_ptr<Detect> detect = std::make_shared<Detect>(80, 3, std::vector<float>({10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0}), 18, true);

        seq->push_back(detect);
    }
    torch::Tensor forward(torch::Tensor x) {
        return torch::cat(x, dimension_)
    }

    torch::nn::Sequential seq = NULL;
};


auto main() -> int {
    std::cout << "hello world!" << std::endl;
}