import torch
import torch.nn as nn
from training.splitting import Splitting
from tokenization.tokening import Tokenization
import torch.nn.functional as F
torch.manual_seed(1337)

class BigramLanguangeModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits,targets)

        return logits,loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits ,loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
    
    def optimizer(self,epochs,batch_size):
        optimizer = torch.optim.AdamW(self.parameters(),lr=1e-3)
        split = Splitting()

        for epoch in range(epochs):

            train_data,val_data = split.data_splitting(entire_text)

            xb,yb = split.get_batch(train_data)
            logits,loss = self(xb,yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(loss.item())

        idx = torch.zeros((1, 1), dtype=torch.long)
        decoded_text = tokens.decoding(self.generate(idx, max_new_tokens=200)[0].tolist())
        print(decoded_text)

with open("data.txt" , "r", encoding="utf-8") as f:
    text = f.read()

vocab_size = len(set(text))

tokens = Tokenization(text)
split = Splitting()

entire_text = tokens.encoding(text) # Tokenizes the entire file
train_data,val_data = split.data_splitting(entire_text)# Train and Val split

m = BigramLanguangeModel(vocab_size)

m.optimizer(epochs=17000,batch_size=32)



"""logits,loss = m(xb,yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1,1),dtype=torch.long)

decoded_text = tokens.decoding(m.genarate(idx,max_new_tokens=100)[0].tolist())
print(decoded_text)"""
