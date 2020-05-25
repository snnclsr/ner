import argparse
import pickle

import torch

from config import UNIQUE_TAGS
from utils import predict_sentence, load_pickle
from crf import CRF
from model import NERTagger

PAD_IDX = 0
UNK_IDX = 1

tag2idx = {tag: idx for idx, tag in enumerate(UNIQUE_TAGS)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

def main():

    arg_parser = argparse.ArgumentParser(description="Named Entity Recognition Testing")
    arg_parser.add_argument("--model_file", required=True, help="Trained model file")
    arg_parser.add_argument("--w2i_file", required=True, help="Word2Index Vocabulary")

    args = arg_parser.parse_args()
    args = vars(args)

    device = "cpu"
    print("Loading the from {}".format(args["model_file"]))
    model = torch.load(args["model_file"], map_location=lambda storage, loc: storage)    
    model = model["model"]
    model.eval()
    # model.device = "cpu"
    print(model)
    model.crf.device = device
    model.device = device

    # model.to(device)

    w2i = load_pickle(args["w2i_file"])
    sentence = (
        "Aziz Nesin 'in yazmış olduğu Nesin Yayınevi tarafından basılan ' Bir Sürgünün Anıları ' "
        "isimli kitap Nesin 'in sürgün yıllarındaki Bursa anılarını anlatıyor ."
    )

    sentence_tokens = sentence.split()
    score, tags = predict_sentence(model, sentence_tokens, w2i, idx2tag)

    for token, tag in zip(sentence_tokens, tags):
        print("{:<15}{:<5}".format(token, tag))


if __name__ == "__main__":
    main()