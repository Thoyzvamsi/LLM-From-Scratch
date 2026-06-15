import torch
import torch.nn as nn
from data_pipeline.data_handling import Data_handling
from data_pipeline.tokenization import Tokenization
import torch.nn.functional as F
torch.manual_seed(1337)

# Hyper parameters
batch_size = 32
block_size = 8
max_interations = 450
lr = 1e-3
eval_iters = 200
n_embd = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokens = Tokenization()
text = tokens.text
vocab_size = len(set(text))

class BigramLanguangeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        logits = self.lm_head(tok_embd) # (B, T, Vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)  # (B*T,c) changes to elongated tensor
            targets = targets.view(B*T)  # It becomes single dimension tensor
            loss = F.cross_entropy(logits,targets)

        return logits,loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        idx = idx.to(device)
        for _ in range(max_new_tokens):
            logits ,loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1).to(device)
        return idx
    
    def optimizer(self,epochs,entire_text):
        optimizer = torch.optim.AdamW(self.parameters(),lr=1e-3) # - lr = 0.003
        split = Data_handling()

        train_data,val_data = split.data_splitting(entire_text)

        for epoch in range(epochs):

            xb,yb = split.get_batch(train_data,batch_size,block_size)

            logits,loss = self(xb,yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(loss.item())

        idx = torch.zeros((1, 1), dtype=torch.long)
        decoded_text = Tokenization().decoding(self.generate(idx, max_new_tokens=200)[0].tolist())
        print(decoded_text)


def main():
    encoded_text = tokens.encoding(text).to(device) # Tokenizes the entire file

    m = BigramLanguangeModel().to(device)

    m.optimizer(max_interations,entire_text=encoded_text)

if __name__ == "__main__" :
    main()