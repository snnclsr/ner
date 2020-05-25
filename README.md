# Named Entity Recognition

This repository implements a Named Entity Recognition system for the Turkish Language. The code will probably work for the other languages as well if the input data format criterion matches.

The model's architecture is Bi-LSTM + CRF. FastText embeddings are used as a pre-trained word embeddings. The code also supports the variable size input.


<!-- Our model is Bi-LSTM + CRF implemented in PyTorch.

Code supports the variable size input. -->

<!-- Code is mainly seperated into 4 files. 

* crf.py

    CRF Module


* model.py

    NER Module

* train.py

    Training Module

* test.py

    Testing Module -->

# Examples
```python
# For the complete example, please see the test.py
sentence = (
    "Aziz Nesin 'in yazmış olduğu Nesin Yayınevi tarafından basılan ' Bir Sürgünün Anıları ' "
    "isimli kitap Nesin 'in sürgün yıllarındaki Bursa anılarını anlatıyor ."
)
# We assume that the sentence is already tokenized.
sentence_tokens = sentence.split()
score, tags = predict_sentence(model, sentence_tokens, w2i, idx2tag)

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

Sentences and the tags should be in separate files. Expected format is:

train_words:

    sent1
    sent2
    sent3
    ...

train_tags:

    tags1
    tags2
    tags3
    ...

We also expect the sentences to be pre-tokenized. The code will split the sentences only from the space.

# Train
```
python train.py -h
-h, --help            show this help message and exit
--train_data TRAIN_DATA [TRAIN_DATA ...]
                    Training data
--valid_data VALID_DATA [VALID_DATA ...]
                    Validation Data
--w2v_file W2V_FILE   Pre-trained Word Embeddings
--hidden_dim HIDDEN_DIM
                    Hidden dimension for the RNN
--num_layers NUM_LAYERS
                    Number of RNN Layers to use
--bidirectional       Option to make the RNNs bidirectional
--dropout_p DROPOUT_P
                    Dropout probability for the embedding layer
--device DEVICE       Device to run the model
--n_epochs N_EPOCHS   Number of epochs to train the model
--model_name MODEL_NAME
                    Model name to save

```

```bash
python train.py --train_data train_words train_tags --valid_data valid_words valid_tags 
--w2v_file <w2v_file> --hidden_dim 64 --num_layers 2 --bidirectional 
--dropout_p 0.3 --device "cuda"
```

# Test

```bash
python test.py -h
usage: test.py [-h] --model_file MODEL_FILE --w2i_file W2I_FILE

Named Entity Recognition Testing

optional arguments:
  -h, --help            show this help message and exit
  --model_file MODEL_FILE
                        Trained model file
  --w2i_file W2I_FILE   Word2Index Vocabulary
```

`w2i_file` is generated from the pre-trained word embeddings vocabulary. You can generate it by using the following code.

```python
index2word = ["<pad>", "<unk>"] + w2v_model.index2word
word2index = {word: index for index, word in enumerate(index2word)}

with open("word2index.pkl", "wb") as f:
    pickle.dump(word2index, f)
```

# FastText Embeddings

FastText embeddings for Turkish can be downloaded from [here (~2.6G)](https://fasttext.cc/docs/en/crawl-vectors.html). The code currently supports only the **`vec`** extension.

# Improvements

* Model saving/loading is handled poorly. There might be better ways to do it (Current version works btw). Switching from GPU to CPU interface is also needs to be handled.
* Allowing the model to be saved/loaded for being able to continue the training later.
* Allowing for different kinds of optimization algorithms and different schedules (Adam is hardcoded right now).
* Different feature extractor models like BERT, ELECTRA, etc.

# References

* [Pytorch Tutorial including CRF](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
* [Turkish BERT (BERTurk)](https://github.com/stefan-it/turkish-bert)