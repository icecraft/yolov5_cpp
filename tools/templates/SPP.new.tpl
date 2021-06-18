

spp_{{seq}} = SPP({{input_channels}}, {{output_channels}},
                    torch::ExpandingArray<2>({{ kernel_size | torch_expanding_array }}),
                    torch::ExpandingArray<2>({{ stride | torch_expanding_array }}),
                    torch::ExpandingArray<2>({{ padding | torch_expanding_array }}),
                    torch::ExpandingArray<2>({{ dilation | torch_expanding_array }}),
                    {{ groups }},
                    {{ bias}},
                    {{ vector<int>({{ pool_kernel_size | cpp_vector_expand }})}}
                );

seq.push_back(spp_{{seq}});