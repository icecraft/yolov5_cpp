
        std::shared_ptr<Detect> detect = std::make_shared<Detect>({{nc}}, {{nl}}, (float []){{ anchor | cpp_vector_expand }}, {{ anchor_len }}, {{ inplace | bool }});

        seq->push_back(detect);