import torch
import torch.nn as nn
from data_pipeline.data_handling import Data_handling
from data_pipeline.tokenization import TokenizerV1
import torch.nn.functional as F
from pathlib import Path
import json
torch.manual_seed(1337)

batch_size = 64
block_size = 256
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_heads = 6
n_layers = 6
dropout = 0.2
tokens = TokenizerV1()
max_new_tokens = 500
text = tokens.text
vocab_size = tokens.vocab_size
weights_path = 'training\weights_optimizer\model_weights.pt'
optimizer_path = 'training\weights_optimizer\optimizer_state.pt'
max_interations = 5
eval_iters = 100
eval_interval = 10

class GPTLanguangeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,n_heads=n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        B ,T = idx.shape
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T,device=device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
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
        for split_name, split_data in {'train': train, 'val': val}.items():
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                X,Y = Data_handling().get_batch(data=split_data,batch_size=batch_size,block_size=block_size)
                logits,loss = self(X,Y)
                losses[i] = loss.item()
            
            out[split_name] = losses.mean()

        self.train()
        return out
    
    @torch.no_grad()
    def generate(self, idx):
        self.load_state_dict(torch.load(weights_path,map_location=device))
        idx = idx.to(device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits ,loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1).to(device)
        result = tokens.decoding(idx[0].tolist())
        return result
            
    def optimizer(self,epochs,entire_text):
        if Path(weights_path).exists():
            # Reloading the weights and opimizer state
            self.load_state_dict(torch.load(weights_path,map_location=device))
            optimizer = torch.optim.AdamW(self.parameters(),lr)
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        else:
            optimizer = torch.optim.AdamW(self.parameters(),lr)

        split = Data_handling()

        train,val = split.data_splitting(entire_text)
        loss_list = []
            
        for epoch in range(epochs):
            if epoch % eval_interval == 0:
                losses = self.loss_function(train,val)
                print(f"Epoch : {epoch} Train Loss {losses['train']:.4f} and Validation loss {losses['val']:.4f}")
            xb,yb = split.get_batch(train,batch_size,block_size)

            logits,loss = self(xb,yb)
            loss_list.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        # Saving the optimizer state and model weights
        torch.save(self.state_dict(), weights_path)
        torch.save(optimizer.state_dict(), optimizer_path)

        idx = torch.zeros((1, 1), dtype=torch.long)
        decoded_text = self.generate(idx)

        with open("data\loss_list.json", "w") as f:
            json.dump(loss_list, f)

        print(decoded_text)

class Block(nn.Module):
    def __init__(self,n_embd,n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttension(n_heads,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class MultiHeadAttension(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads] , dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Self attention block
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.query = nn.Linear(n_embd ,head_size ,bias=False)
        self.key = nn.Linear(n_embd ,head_size ,bias=False)
        self.value = nn.Linear(n_embd ,head_size ,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        B,T,C = x.shape
        q = self.query(x) # (B,T,C)
        k = self.key(x) 
        v = self.value(x) 
        wei = q @ k.transpose(-2,-1) * (C // n_heads) ** -0.5 # (B,T,C) @ (B,C,T) = # (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0 ,float('-inf')) #
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # (B,T,T) @ (B,T,C) = # (B,T,C)

        return out
    
class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4 , n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)


def main():
    encoded_text = tokens.encoding(text).to(device) # Tokenizes the entire file
    print(f"encoded_text length: {len(encoded_text)}")

    m = GPTLanguangeModel().to(device)

    m.optimizer(max_interations,entire_text=encoded_text)

if __name__ == "__main__" :
    main()