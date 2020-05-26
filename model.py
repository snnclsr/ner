from typing import List, Optional

from utils import to_tensor, generate_sent_masks
from config import PAD_IDX

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from crf import CRF

class NERTagger(nn.Module):
    """
    NER network.
    hidden_size: lstm hidden_size
    output_size: total number of tags
    num_layers: Number of RNN layers
    bidirectional: RNN will be bidirectional or not?
    weights: Pretrained word embeddings

    embedding layer will fill the pad idx with padding vector.
    """
    
    def __init__(self, hidden_size: int, output_size: int, num_layers: int=1, 
                 bidirectional: bool=False, dropout_p: float=0.1,  
                 device: str="cpu", weights: Optional=None, num_embeddings: Optional=None, 
                 embedding_dim: Optional=None):
        super(NERTagger, self).__init__()
        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(weights, padding_idx=PAD_IDX)
        else:
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, 
                                          embedding_dim=embedding_dim, 
                                          padding_idx=PAD_IDX)
  
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        self.device = device
        self.dropout = nn.Dropout(p=dropout_p)
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim, 
                            hidden_size=hidden_size, 
                            bidirectional=bidirectional, 
                            num_layers=num_layers,
                            batch_first=True)
        if self.bidirectional:
            hidden_size = 2 * hidden_size
        self.crf = CRF(hidden_size, output_size, device=device)
        
    def loss(self, x: List[List], tags: List[List]):
        features, mask = self.feature_extraction(x)
        tags = to_tensor(tags, device=self.device)
        loss = self.crf.loss(features, tags, mask)
        return loss
        
    def feature_extraction(self, x: List[List]):
        # Batch_iter sorts the sentences.
        sents_tensor = to_tensor(x, device=self.device) # (bs, max_seq_len)
        seq_lengths = torch.tensor([len(sent) for sent in x])
        masks = generate_sent_masks(sents_tensor, seq_lengths).to(self.device)
        x = self.embedding(sents_tensor)
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths=seq_lengths, batch_first=True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        return x, masks

    def forward(self, x: List[List]):
        features, masks = self.feature_extraction(x)
        scores, tags = self.crf(features, masks)
        return scores, tags