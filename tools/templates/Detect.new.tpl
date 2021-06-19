
detect = Detect({{nc}}, {{nl}}, vector<float32>({{ anchor | cpp_vector_expand }}), {{ anchor_len }}, {{ inplace | bool }});

seq.push_back(detect);