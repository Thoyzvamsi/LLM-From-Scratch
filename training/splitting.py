import torch
from tokenization import tokening

class Splitting:
    def data_splitting(self,data):
        n = 0.9
        # -- data splitting -- 
        train_data = data[:int(len(data)*n)]
        val_data = data[int(len(train_data)*n) : ]

        return train_data , val_data
    
    def get_batch(self,data):
        torch.manual_seed(1334)
        block_size = 4
        batch_size = 8

        ix = torch.randint(len(data)-block_size,(batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x,y

