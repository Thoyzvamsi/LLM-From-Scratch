import torch
from tokenization import tokening
import torch.nn as nn
from torch import functional as F

class Training_NN:
    def __init__(self,text,vocab_size):
        self.data = text
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def data_splitting(self):
        n = 0.9
        # -- data splitting -- 
        train_data = self.data[:len(self.data)*n]
        val_data = self.data[len(train_data) : ]

        return train_data , val_data
    
    def get_batch(self,data):
        torch.manual_seed(1334)
        block_size = 4
        batch_size = 8

        ix = torch.randint(len(data)-block_size,(batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x,y
    
    def foward(self, idx, targets):
        logits = self.token_embedding_table(idx) # (B,T,C)
        B,T,C = logits.shape()
        logits = logits.view(B*T,C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits,targets)

        return logits,loss


with open("data.txt" , "r", encoding="utf-8") as f:
    text = f.read()

vocab_size = len(set(text))
tokens = tokening.Tokenization(text,vocab_size)
encoded_text = tokens.encoding(text) # Tokenizes the entire file
train = Training_NN(text)
train_data,val_data = train.data_splitting()
