import argparse
import pickle

import torch

PAD_IDX = 0
UNK_IDX = 1

unique_tags = ["PAD", "O", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-DATE", "I-DATE"]
tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

from utils import predict_sentence, load_pickle
from crf import CRF
from model import NERTagger

def main():
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model_path = "models/model.pth"
    print("Loading the from {}".format(model_path))
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = model["model"]
    model.device = "cpu"
    model.to(device)
    print(model)

    w2i_fn = "word2index.pkl"

    w2i = load_pickle(w2i_fn)
    sentence = "Aziz Nesin 'in yazmış olduğu Nesim Yayınevi tarafından basılan ' Bir Sürgünün Anıları ' isimli kitap Nesin 'in sürgün yıllarındaki Bursa anılarını anlatıyor ."
    
    score, tags = predict_sentence(model, sentence, w2i, device=device)
    print("Sentence: {}".format(sentence))
    print("Score: {}".format(score))
    print("Tags: {}".format(tags))


if __name__ == "__main__":
    main()