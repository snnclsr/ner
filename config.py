# Some configuration variables.

PAD_IDX = 0 # Padding index
UNK_IDX = 1 # Unknown word index

# Tags for the named entities.
UNIQUE_TAGS = [
    "PAD", "O", "B-ORG", "I-ORG", 
    "B-LOC", "I-LOC", "B-PER", "I-PER"
]
tag2idx = {tag: idx for idx, tag in enumerate(UNIQUE_TAGS)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}
