# Named Entity Recognition
Turkish Named Entity Recognition

# Examples
```python
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = model["model"]
    model.device = "cpu"
    print(model)

    w2i_fn = "word2index.pkl" # Vocabulary

    w2i = load_pickle(w2i_fn)
    sentence = (
        "Aziz Nesin 'in yazmış olduğu Nesin Yayınevi tarafından basılan ' Bir Sürgünün Anıları ' "
        "isimli kitap Nesin 'in sürgün yıllarındaki Bursa anılarını anlatıyor ."
    )
    # We assume that the sentence is already tokenized.
    sentence_tokens = sentence.split()
    score, tags = predict_sentence(model, sentence_tokens, w2i, idx2tag, device=device)
    
    for token, tag in zip(sentence_tokens, tags):
        print("{:<15}{:<5}".format(token, tag))

    # Output:
    # Aziz           B-PER
    # Nesin          I-PER
    # 'in            O
    # yazmış         O
    # olduğu         O
    # Nesin          B-ORG
    # Yayınevi       I-ORG
    # tarafından     O
    # basılan        O
    # '              O
    # Bir            O
    # Sürgünün       O
    # Anıları        O
    # '              O
    # isimli         O
    # kitap          O
    # Nesin          B-PER
    # 'in            O
    # sürgün         O
    # yıllarındaki   O
    # Bursa          B-LOC
    # anılarını      O
    # anlatıyor      O
    # .              O
```


# Data Format

Expected data format:

    In seperate files


# Train

```bash
python train.py --train_data <train_data> --valid_data <valid_data> --w2v_file <w2v_file> --hidden_dim 64 --num_layers 2 --bidirectional --dropout_p 0.3 --device "cuda"
```

# Test

```

```

# Improvements

* Model saving/loading is handled poorly. There might be better ways to do it (Current version works btw). Also changing from GPU to CPU interface is also need to be handled.
* Allowing the model to be saved/loaded for being able to continue the training later.
* Allowing for different kind of optimization algorithms and different schedules (Adam is hardcoded right now).
* Different feature extractor models like BERT, ELECTRA etc.
