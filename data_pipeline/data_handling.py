import torch

class Data_handling:
    def data_splitting(self,data=None):
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
    
    
    def data_quality_filter(self, text):
        if len(text) < 400 or len(text.split()) < 100:
            return False
        
        alpha_ratio = sum(c.isalpha() for c in text)/len(text)

        # "Hello12389" ratio = 5(chars) / 10(whole word) < 0.6
        if alpha_ratio < 0.6:
            return False
        
        # Same as the above but for sentences
        sentences = text.split('.')
        uniquness = len(set(sentences))/len(sentences)
        if uniquness < 0.4:
            return False
        
        return True 