import numpy as np
import random
import unicodedata
import re

import torchtext
import torch

def load_pretrained_embedding(pretrained_path: str, vocab: torchtext.vocab.Vocab, num_embedding: int):
    print("Load pre-trained embedding")
    embedding_tensor = torch.zeros((len(vocab), num_embedding))
    word2embedding = dict()
    
    with open(pretrained_path, 'rb') as file:
        for line in file:
            line = line.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            word2embedding[word] = vect
    
    for word in vocab.itos:
        idx = vocab[word]
        unk_tensor = np.random.normal(scale=0.6, size=(num_embedding,))
        word_tensor = word2embedding.get(word, unk_tensor)
        embedding_tensor[idx, :] = torch.tensor(word_tensor)
    
    print(f"Sucessfully get pretrained embedding size: {embedding_tensor.shape}")
    return embedding_tensor
    

def random_seed(seed: int=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s