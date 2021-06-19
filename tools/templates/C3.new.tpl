
        c3_{{seq}} = C3({{in_channels}}, {{out_channels}}, 
                          torch::ExpandingArray<2>({{ kernel_size | torch_expanding_array }}),
                          torch::ExpandingArray<2>({{ stride | torch_expanding_array }}),
                          torch::ExpandingArray<2>({{ padding | torch_expanding_array }}),
                          torch::ExpandingArray<2>({{ dilation | torch_expanding_array }}),
                          {{ groups }},
                          {{ bias | bool }},
                          {{ n }},
                          {{ shortcut | bool }},
                          {{ groups2 }},
                          {{ eps }}
                            );
                            
        seq.push_back(c3_{{seq}});