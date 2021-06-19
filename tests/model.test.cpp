
#include <torch/torch.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>


struct Model {
    Model () {   
        
        focus_1 = Focus(12, 32, 
                              torch::ExpandingArray<2>({3, 3}),
                              torch::ExpandingArray<2>({1, 1}),
                              torch::ExpandingArray<2>({1, 1}),
                              torch::ExpandingArray<2>({1, 1}),
                              1,
                              false
                            );
                            
        seq.push_back(focus_1);
        conv_1 = Conv(32, 64, 
                          torch::ExpandingArray<2>({3, 3}),
                          torch::ExpandingArray<2>({2, 2}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq.push_back(conv_1);
        c3_1 = C3(64, 32, 
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
                            
        seq.push_back(c3_1);
        conv_2 = Conv(64, 128, 
                          torch::ExpandingArray<2>({3, 3}),
                          torch::ExpandingArray<2>({2, 2}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq.push_back(conv_2);
        c3_2 = C3(128, 64, 
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
                            
        seq.push_back(c3_2);
        conv_3 = Conv(128, 256, 
                          torch::ExpandingArray<2>({3, 3}),
                          torch::ExpandingArray<2>({2, 2}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq.push_back(conv_3);
        c3_3 = C3(256, 128, 
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
                            
        seq.push_back(c3_3);
        conv_4 = Conv(256, 512, 
                          torch::ExpandingArray<2>({3, 3}),
                          torch::ExpandingArray<2>({2, 2}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq.push_back(conv_4);
        spp_1 = SPP(512, 256,
                    torch::ExpandingArray<2>({1, 1}),
                    torch::ExpandingArray<2>({1, 1}),
                    torch::ExpandingArray<2>({0, 0}),
                    torch::ExpandingArray<2>({1, 1}),
                    1,
                    false,
                    vector<int>({5, 9, 13})
                    );

        seq.push_back(spp_1);
        c3_4 = C3(512, 256, 
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
                            
        seq.push_back(c3_4);
        conv_5 = Conv(512, 256, 
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({0, 0}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq.push_back(conv_5);
        concat_1 = Concat(1);

        seq.push_back(concat_1);
        c3_5 = C3(512, 128, 
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
                            
        seq.push_back(c3_5);
        conv_6 = Conv(256, 128, 
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({0, 0}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq.push_back(conv_6);
        concat_2 = Concat(1);

        seq.push_back(concat_2);
        c3_6 = C3(256, 64, 
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
                            
        seq.push_back(c3_6);
        conv_7 = Conv(128, 128, 
                          torch::ExpandingArray<2>({3, 3}),
                          torch::ExpandingArray<2>({2, 2}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq.push_back(conv_7);
        concat_3 = Concat(1);

        seq.push_back(concat_3);
        c3_7 = C3(256, 128, 
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
                            
        seq.push_back(c3_7);
        conv_8 = Conv(256, 256, 
                          torch::ExpandingArray<2>({3, 3}),
                          torch::ExpandingArray<2>({2, 2}),
                          torch::ExpandingArray<2>({1, 1}),
                          torch::ExpandingArray<2>({1, 1}),
                          1,
                          false
                            );
                            
        seq.push_back(conv_8);
        concat_4 = Concat(1);

        seq.push_back(concat_4);
        c3_8 = C3(512, 256, 
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
                            
        seq.push_back(c3_8);
        detect = Detect(80, 3, vector<float>({10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0}), 18, true);

        seq.push_back(detect);
    }
    torch::Tensor forward(torch::Tensor x) {
        return torch::cat(x, dimension_)
    }
};