import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pickle

from gensim.models import KeyedVectors

import numpy as np
import torch
from torch import Tensor


PAD_IDX = 0
UNK_IDX = 1
unique_tags = ["PAD", "O", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-DATE", "I-DATE"]
tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

def predict_sentence(model, sentence, w2i, i2t, device="cpu"):

    encoded_sent = encode_sent(sentence, w2i)  
    score, tags = model([encoded_sent])
    tags = decode_tags(tags[0], i2t)
    return score.item(), tags


def load_pickle(filename):
    
    with open(filename, "rb") as f:
        data = pickle.load(f)

    return data


def load_data(filename: str):
    """
    Load the data from the given filename.
    """
    with Path(filename).open('r', encoding="utf-8") as f:
        data = f.read().split('\n')

    return data


def strip_sents_and_tags(sents: List, tags: List):

    tmp_train_sents, tmp_train_tags = [], []
    for sent, tag in zip(sents, tags):
        if sent.strip():
            tmp_train_sents.append(sent.strip()) 
            tmp_train_tags.append(tag.strip())
    
    return tmp_train_sents, tmp_train_tags


def encode_tags(sent_tags: List, tag2idx: Dict):
    """
    Replace the tags (O, B-LOC etc.) with the corresponding idx from the 
    tag2idx dictionary
    """
    encoded_tags = [tag2idx[token_tag] for token_tag in sent_tags]
    return encoded_tags


def decode_tags(tags: List, idx2tag: Dict):
    """
    Decode the tags by replacing the tag indices with the original tags.
    """
    decoded_tags = [idx2tag[tag_idx] for tag_idx in tags] # if tag_idx != PAD_IDX
    return decoded_tags


def load_wv(filename: str, limit: Optional[int]=None) -> KeyedVectors:
    """
    Load the fastText pretrained word embeddings from given filename.
    """
    embeddings = KeyedVectors.load_word2vec_format(filename, 
                                            binary=False, 
                                            limit=limit, 
                                            unicode_errors='ignore')
    return embeddings


def encode_sent(sent_tokens: List, word2idx: Dict) -> List:
    """
    Replace the tokens with the corresponding index from `word2idx`
    dictionary. 
    """
    encoded_sent = [word2idx.get(token, UNK_IDX) for token in sent_tokens]
    return encoded_sent


def decode_sent(sent: List, idx2word: Dict) -> List:
    """
    Decode the sentence to the original form by replacing token indices 
    with the words.
    """
    decoded_sent = [idx2word[token_idx] for token_idx in sent if token_idx != PAD_IDX]
    return decoded_sent


def pad_sequences(sequences: List[List], pad_idx: Optional[int]=0) -> List[List]:
    """
    Pad the sequences to the maximum length sequence.
    """
    max_len = max([len(seq) for seq in sequences])
    
    padded_sequence = []
    for seq in sequences:
        seq_len = len(seq)
        pad_len = max_len - seq_len
        padded_seq = seq + [pad_idx] * pad_len
        padded_sequence.append(padded_seq)
    
    return padded_sequence


def to_tensor(sents: List[List], device: str="cpu") -> Tensor:
    """
    Pad the sentences and convert them to the torch tensor.
    """
    padded_sents = pad_sequences(sents)
    sent_tensor = torch.tensor(padded_sents, dtype=torch.long, device=device)
    return sent_tensor # (batch_size, max_seq_len)


def generate_sent_masks(sents: Tensor, lengths: Tensor) -> Tensor:
    """
    Generate the padding masking for given sents from lenghts. 
    Assumes lengths are sorted by descending order (batch_iter provides this).
    """
    max_len = lengths[0]
    bs = sents.shape[0]
    mask = torch.arange(max_len).expand(bs, max_len) < lengths.unsqueeze(1)
    return mask.byte()


def batch_iter(data: List[List], batch_size: int, shuffle: bool=False) -> Tuple[List, List]:
    
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    
    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i+1) * batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        sents = [e[0] for e in examples]
        tags = [e[1] for e in examples]
        yield sents, tags

