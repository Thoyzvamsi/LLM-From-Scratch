import torch
import pickle

class Tokenization:
    def __init__(self):

        with open("tokenizer.pkl", "rb") as f:
            data = pickle.load(f)

        self.vocab_size = data["vocab_size"]
        self.merges = data["merges"]
        self.vocab = data["vocab"]
#decodes tokens to text
    def decode(self,ids):
        decoded=b"".join(self.vocab[idx] for idx in ids)
        return decoded.decode("utf-8",errors="replace")
#encodes text to tokens 
    def encode(self,text): # we have to convert given text into unicode then merge repeated pairs 
        encoded=list(text.encode("utf-8",errors="replace"))
        while len(encoded) >= 2:
            stats = self.get_stat(encoded)
            pair = min(stats ,key=lambda p: self.merges.get(p,float('inf')))
            if pair not in self.merges:
                break # no pair is repeating
            idx=self.merges[pair]
            encoded=self.merge(encoded,pair,idx)
        encoded = torch.tensor(encoded)
        return encoded
