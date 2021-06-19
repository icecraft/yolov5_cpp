
    detect = Detect({{nc}}, {{nl}}, vector<int>({{ anchor | cpp_vector_expand }}), {{ anchor_len }}, {{ inplace }});

    seq.push_back(detect);