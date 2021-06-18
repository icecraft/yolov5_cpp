

  c3_{{seq}} = C3({{input_channels}}, {{output_channels}}, 
                        torch::ExpandingArray<2>({{ kernel_size | torch_expanding_array }}),
                        torch::ExpandingArray<2>({{ stride | torch_expanding_array }}),
                        torch::ExpandingArray<2>({{ padding | torch_expanding_array }}),
                        torch::ExpandingArray<2>({{ dilation | torch_expanding_array }}),
                        {{ groups }},
                        {{ bias}},
                        {{ n }},
                        {{ shortcut }},
                        {{ g}},
                        {{ eps }}
                      );
                      
  seq.push_back(c3_{{seq}});