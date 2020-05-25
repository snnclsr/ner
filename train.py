import argparse
import logging
import time
import datetime
import math

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from seqeval.metrics import f1_score

from utils import load_data, strip_sents_and_tags
from utils import encode_sent, encode_tags, load_wv, batch_iter, decode_tags
from config import UNIQUE_TAGS, PAD_IDX, idx2tag, tag2idx
from model import NERTagger


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def generate_tags(model, data, batch_size=32, device="cpu"):
    """
    Generate the tags (predictions) for the samples in the data.
    """
    all_decoded_targets = []
    all_decoded_preds = []
    
    for batch in batch_iter(data, batch_size=batch_size, shuffle=False):
        # batch = (b.to(device) for b in batch)        
        sents, tags = batch
        scores, pred_tags = model(sents)
        len_test_tags = [len(test_tag) for test_tag in tags]
        cleaned_test_preds = [pred[:l] for l, pred in zip(len_test_tags, pred_tags)]
        
        gt_tags = [decode_tags(tag, idx2tag) for tag in tags]
        pred_tags = [decode_tags(tag, idx2tag) for tag in cleaned_test_preds]
        
        all_decoded_targets.extend(gt_tags)
        all_decoded_preds.extend(pred_tags)
        
    return all_decoded_targets, all_decoded_preds


def train_step(model, loss_fn, optimizer, train_data, batch_size=32, device="cpu"):
    """
    Train the model for 1 epoch.
    """
    total_loss = 0.0
    model.train()
    start_time = time.time()
    total_step = math.ceil(len(train_data) / batch_size)
    
    for step, batch in enumerate(batch_iter(train_data, batch_size=batch_size, shuffle=True)):
        if step % 250 == 0 and not step == 0:
            elapsed_since = time.time() - start_time
            logger.info("Batch {}/{}\tElapsed since: {}".format(step, total_step, 
                                                          str(datetime.timedelta(seconds=round(elapsed_since)))))
        # batch = (b.to(device) for b in batch)
        sents, tags = batch
        optimizer.zero_grad()
        train_loss = model.loss(sents, tags)
        total_loss += train_loss.item()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
            
    avg_train_loss = total_loss / total_step
    return avg_train_loss
        
                
def eval_step(model, loss_fn, data, batch_size=32, device="cpu"):
    """
    Evaluate the model for the given data_loader.
    """
    total_loss = 0.0
    model.eval()
    total_step = math.ceil(len(data) / batch_size)

    for batch in batch_iter(data, batch_size=batch_size, shuffle=False):
        # batch = (b.to(device) for b in batch)
        sents, tags = batch
        eval_loss = model.loss(sents, tags)
        total_loss += eval_loss.item()
        
    average_eval_loss = total_loss / total_step
    return average_eval_loss
    

def train(model, loss_fn, optimizer, train_dl, valid_dl, n_epochs=1, device="cpu"):
    """
    Training loop.
    """
    print("...Training for {} epochs...".format(n_epochs))
    print("Number of training samples: ", len(train_dl))
    train_losses = []
    if valid_dl is not None:
        valid_losses = []
        
    for epoch in range(n_epochs):
        
        start_time = time.time()
        
        train_loss = train_step(model, loss_fn, optimizer, train_dl, device=device)
        train_losses.append(train_loss)
        
        elapsed_time = time.time() - start_time
        logger.info("Epoch {}/{} is done. Took: {} Loss: {:.5f}".format(epoch+1, 
                                                                n_epochs, 
                                                                str(datetime.timedelta(seconds=round(elapsed_time))), 
                                                                train_loss))
        
        valid_loss = eval_step(model, loss_fn, valid_dl, device=device)
        valid_losses.append(valid_loss)
        print("Validation Loss: {:.5f}".format(valid_loss))
        val_targets, val_preds = generate_tags(model, valid_dl, device=device)
        print("Validation f1-score: ", f1_score(val_targets, val_preds, average="macro"))
            
        print("=" * 50)
    
    return train_losses, valid_losses



def main():
    arg_parser = argparse.ArgumentParser(description="Named Entity Recognition Training")
    arg_parser.add_argument("--train_data", required=True, help="Training data", nargs='+')
    arg_parser.add_argument("--valid_data", required=True, help="Validation Data", nargs='+')
    arg_parser.add_argument("--w2v_file", help="Pretrained Word Embeddings")
    arg_parser.add_argument("--hidden_dim", type=int, default=32, 
                            help="Hidden dimension for the RNN")
    arg_parser.add_argument("--num_layers", type=int, default=1, 
                            help="Number of RNN Layers to use")
    arg_parser.add_argument("--bidirectional", action="store_true", 
                            help="Option to make RNNs bidirectional")
    arg_parser.add_argument("--dropout_p", type=float, default=0.1, 
                            help="Dropout probability for the embedding layer")
    arg_parser.add_argument("--device", type=str, default="cpu", 
                            help="Device to run the model")
    arg_parser.add_argument("--n_epochs", type=int, default=1, 
                            help="Number of epochs to train the model")
    arg_parser.add_argument("--model_name", type=str, default="model.pth", 
                            help="Model name to save")
    # arg_parser.add_argument("")

    args = arg_parser.parse_args()
    args = vars(args)
    print(args)

    device = "cuda" if args["device"] == "cuda" else "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and args["device"] == "cuda":
        logger.info("Device is specified as cuda. But there is no cuda device available in your system.")
        exit(0)

    train_sents = load_data(args["train_data"][0])
    train_tags = load_data(args["train_data"][1])
    
    valid_sents = load_data(args["valid_data"][0])
    valid_tags = load_data(args["valid_data"][1])

    train_sents, train_tags = strip_sents_and_tags(train_sents, train_tags)
    valid_sents, valid_tags = strip_sents_and_tags(valid_sents, valid_tags)
    
    # Split the space seperated sents. Assuming that the sentences are tokenized.
    train_sents = [sent.split() for sent in train_sents]
    valid_sents = [sent.split() for sent in valid_sents]

    # Split the space seperated tags.
    train_tags = [tags.split() for tags in train_tags]
    valid_tags = [tags.split() for tags in valid_tags]

    logger.info(f"Total train sents/tags: {len(train_sents)}/{len(train_tags)}")
    logger.info(f"Total valid sents/tags: {len(valid_sents)}/{len(valid_tags)}")

    # Replace the tags with the indices
    tag2idx = {tag: idx for idx, tag in enumerate(UNIQUE_TAGS)}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    train_tags_idx = [encode_tags(tags, tag2idx) for tags in train_tags]
    valid_tags_idx = [encode_tags(tags, tag2idx) for tags in valid_tags]

    # Load the pretrained word embeddings.
    w2v_fn = args["w2v_file"]
    logger.info(f"Loading the pretrained word embeddings from {w2v_fn}")

    word_vectors = load_wv(args["w2v_file"], limit=100)
    # We will add 2 additional vectors for the padding & unknown tokens.
    # padding_idx will be the first index of the word vector matrix.
    # unknown_idx will be the second index of the word vector matrix.
    additional_vectors = np.zeros(shape=(2, 300))

    index2word = ["<pad>", "<unk>"] + word_vectors.index2word
    word2index = {word: index for index, word in enumerate(index2word)}
    weights = np.concatenate((additional_vectors, word_vectors.vectors))

    weights = torch.from_numpy(weights).float()
    # embedding = nn.Embedding.from_pretrained(weights, padding_idx=0)
    # embedding(torch.LongTensor([2])) # embeddings for token '.'

    # Replace the sent_tokens with the indices from word2index.
    train_sents_idx = [encode_sent(sent, word2index) for sent in train_sents]
    valid_sents_idx = [encode_sent(sent, word2index) for sent in valid_sents]

    # Final form of the data
    """
    [
        [
            [sent1_token1_idx, sent1_token2_idx, sent1_token3_idx, ...], 
            [sent_1_tag1_idx, sent1_tag2_idx, sent1_tag3_idx, ...]
        ],
        [
            [sent2_token1_idx, sent2_token2_idx, sent2_token3_idx, ...], 
            [sent2_tag1_idx, sent2_tag2_idx, sent2_tag3_idx, ...]
        ],
        ...
    ]
    """
    train_data = list(zip(*[train_sents_idx, train_tags_idx]))
    valid_data = list(zip(*[valid_sents_idx, valid_tags_idx]))

    model = NERTagger(
        hidden_size=args["hidden_dim"],
        output_size=len(UNIQUE_TAGS),
        num_layers=args["num_layers"],
        bidirectional=args["bidirectional"],
        dropout_p=args["dropout_p"],
        weights=weights,
        device=args["device"]
    )
    model.to(device)
    print(model)

    # Defining the optimizer
    optimizer = optim.Adam(model.parameters())
    # Defining the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    train_losses, valid_losses = train(model, criterion, optimizer, train_data, valid_dl=valid_data, n_epochs=args["n_epochs"], device=device)

    logger.info("Saving the trained model to the {}".format(args["model_name"]))
    params = {
        "model": model
    }
    torch.save(params, args["model_name"])




if __name__ == "__main__":
    main()