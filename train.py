import argparse
import logging

from utils import load_data, strip_sents_and_tags
from utils import encode_tags, load_wv
from config import UNIQUE_TAGS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def main():
    arg_parser = argparse.ArgumentParser(description="Named Entity Recognition Training")
    arg_parser.add_argument("--train_data", required=True, help="Training data", nargs='+')
    arg_parser.add_argument("--valid_data", help="Validation Data", nargs='+')
    arg_parser.add_argument("--w2v_file", help="Pretrained Word Embeddings")

    args = arg_parser.parse_args()
    args = vars(args)
    print(args)
    # dataset_fn = Path("/home/sinan/learning/papers/turkish_nlp/ner/data/")
    # datas = {}


    # for file in dataset_fn.iterdir():
    #     print(file.name)
    #     datas[file.name] = load_data(file)

    
    train_sents = load_data(args["train_data"][0])
    train_tags = load_data(args["train_data"][1])
    
    valid_sents = load_data(args["valid_data"][0])
    valid_tags = load_data(args["valid_data"][1])

    train_sents, train_tags = strip_sents_and_tags(train_sents, train_tags)
    valid_sents, valid_tags = strip_sents_and_tags(valid_sents, valid_tags)
    
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
    logger.info(f"Loading the pretrained word embeddings from {args["w2v_file"]}")

    word_vectors = load_wv(args["w2v_file"], limit=100)


if __name__ == "__main__":
    main()