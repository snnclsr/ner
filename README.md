# ner
Turkish Named Entity Recognition

# Data Format

# Train

```
python train.py --train_data <train_data> --valid_data <valid_data> --w2v_file <w2v_file> --hidden_dim 64 --num_layers 2 --bidirectional --dropout_p 0.3 --device "cuda"
```

# Test

```

```

# Improvements

* Model saving/loading is handled poorly. There might be better ways to do it (Current version works btw). 
* Allowing the model to be saved/loaded for being able to continue the training later.
* Allowing for different kind of optimization algorithms and different schedules (Adam is hardcoded right now).
* Different feature extractor models like BERT, ELECTRA etc.
