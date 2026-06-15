import torch
import torch.nn as nn
from data_pipeline.data_handling import Data_handling
from data_pipeline.tokenization import Tokenization
import torch.nn.functional as F
torch.manual_seed(1337)


class BigramLanguangeModel(nn.Module):
    def __init__(self,vocab_size,device="cpu"):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
        self.device = device

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C)

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
        idx = idx.to(self.device)
        for _ in range(max_new_tokens):
            logits ,loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1).to(self.device)
        return idx
    
    def optimizer(self,epochs,entire_text):
        optimizer = torch.optim.AdamW(self.parameters(),lr=1e-3) # - lr = 0.003
        split = Data_handling()

        train_data,val_data = split.data_splitting(entire_text)
        device = self.device

        for epoch in range(epochs):

            xb,yb = split.get_batch(train_data)

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
    tokens = Tokenization()
    text = tokens.text
    vocab_size = len(set(text))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoded_text = tokens.encoding(text).to(device) # Tokenizes the entire file

    m = BigramLanguangeModel(vocab_size , device=device).to(device)

    m.optimizer(epochs=60000,entire_text=encoded_text)

if __name__ == "__main__" :
    main()