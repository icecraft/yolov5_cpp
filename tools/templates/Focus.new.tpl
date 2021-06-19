

  focus_{{seq}} = Focus({{in_channels}}, {{out_channels}}, 
                        torch::ExpandingArray<2>({{ kernel_size | torch_expanding_array }}),
                        torch::ExpandingArray<2>({{ stride | torch_expanding_array }}),
                        torch::ExpandingArray<2>({{ padding | torch_expanding_array }}),
                        torch::ExpandingArray<2>({{ dilation | torch_expanding_array }}),
                        {{ groups }},
                        {{ bias}}
                      );
                      
  seq.push_back(focus_{{seq}});





