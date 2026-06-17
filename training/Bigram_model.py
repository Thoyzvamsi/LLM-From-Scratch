import torch
import torch.nn as nn
from data_pipeline.data_handling import Data_handling
from data_pipeline.tokenization import Tokenization
import torch.nn.functional as F
torch.manual_seed(1337)

# Hyper parameters
batch_size = 32
block_size = 16
max_interations = 20000
lr = 1e-3
eval_iters = 200
n_embd = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 100
tokens = Tokenization()
text = tokens.text
vocab_size = len(set(text))


class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,n_embd),
            nn.ReLU()
        )

    def forward(self,x):
        return self.net(x)
    

class BigramLanguangeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.sa_heads = MultiHeadAttension(4,n_embd//4)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        B ,T = idx.shape 
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T,device=device))
        x = tok_embd + pos_embd
        x = self.sa_heads(x)
        x = self.ffwd(x)
        logits = self.lm_head(x) # (B, T, Vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)  # (B*T,c) changes to elongated tensor
            targets = targets.view(B*T)  # It becomes single dimension tensor
            loss = F.cross_entropy(logits,targets)

        return logits,loss
    
    @torch.no_grad()
    def loss_function(self,train,val):
        out = {}
        self.eval()
        k = 0
        for split in [train,val]:
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                X,Y = Data_handling().get_batch(data=split,batch_size=batch_size,block_size=block_size)
                logits,loss = self(X,Y)
                losses[i] = loss.item()
            if k == 0:
                out['train'] = losses.mean()
                k = 1
            else:
                out['val'] = losses.mean()
                k = 0

        self.train()
        return out
            
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        idx = idx.to(device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits ,loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1).to(device)
        return idx
    
    def optimizer(self,epochs,entire_text):
        optimizer = torch.optim.AdamW(self.parameters(),lr=1e-3) # - lr = 0.003
        split = Data_handling()

        train,val = split.data_splitting(entire_text)

        for epoch in range(epochs):
            if epoch % eval_interval == 0:
                losses = self.loss_function(train,val)
                print(f"Train Loss {losses['train']:.4f} and Validation loss {losses['val']:.4f}")
            xb,yb = split.get_batch(train,batch_size,block_size)

            logits,loss = self(xb,yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        idx = torch.zeros((1, 1), dtype=torch.long)
        decoded_text = Tokenization().decoding(self.generate(idx, max_new_tokens=200)[0].tolist())
        print(decoded_text)


# Self attention block
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.query = nn.Linear(n_embd ,head_size ,bias=False)
        self.key = nn.Linear(n_embd ,head_size ,bias=False)
        self.value = nn.Linear(n_embd ,head_size ,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    
    def forward(self,x):
        B,T,C = x.shape
        q = self.query(x) # (B,T,C)
        k = self.key(x) # (B,T,C)
        v = self.value(x) # (B,T,C)
        wei = q @ k.transpose(-2,-1) * C ** -0.5 # (B,T,C) @ (B,C,T) = # (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0 ,float('-inf')) #
        wei = F.softmax(wei,dim=-1) 
        out = wei @ v # (B,T,T) @ (B,T,C) = # (B,T,C)

        return out
    
class MultiHeadAttension(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self,x):
        return torch.cat([h(x) for h in self.heads] , dim=-1)

def main():
    encoded_text = tokens.encoding(text).to(device) # Tokenizes the entire file

    m = BigramLanguangeModel().to(device)

    m.optimizer(max_interations,entire_text=encoded_text)

if __name__ == "__main__" :
    main()