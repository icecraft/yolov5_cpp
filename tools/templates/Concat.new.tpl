
        std::shared_ptr<Concat> concat_{{seq}} = std::make_shared<Concat>({{dimension}});

        seq->push_back(concat_{{seq}});