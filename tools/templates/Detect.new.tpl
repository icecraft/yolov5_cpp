
        detect = Detect({{nc}}, {{nl}}, vector<float>({{ anchor | cpp_vector_expand }}), {{ anchor_len }}, {{ inplace | bool }});

        seq->push_back(detect);