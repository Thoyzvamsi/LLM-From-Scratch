import torch

# class Tokenization:
#     def __init__(self):
#         self.text = open("data\data.txt",'r').read()
#         self.chars = sorted(list(set(self.text)))
#         self.vocab_size = len(set(self.text))

#     # -- Encoding to Tokens --
#     def encoding(self,context_letters):
#         tokens = []
#         for letter in context_letters:
#             num = self.chars.index(letter)
#             tokens.append(num)
#         tokens = torch.tensor(tokens)
#         return tokens
    
#     # -- Decoding to text -- 
#     def decoding(self,context_numbers):
#         decoded_text = ""
#         for number in context_numbers:
#             decoded_text = decoded_text+self.chars[number]
#         return decoded_text

# data pre processing
with open("data/data.txt","r",encoding="utf-8") as f:
    raw_text=f.read()

def get_stat(ids): #gives dictionary of token pair:number of times repeated
    count={}
    for pair in zip(ids,ids[1:]):
        count[pair]=count.get(pair,0)+1
    return count
# stats=get_stat(tokens)
# print(sorted(((v,k) for k,v in stats.items()),reverse=True))
# top_pair=max(stats,key=stats.get)
def merge(ids,pair,idx): #merges most repeated pairs into single index values
    i=0
    new_ids=[]
    while i<len(ids):
        if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
            new_ids.append(idx)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids
# merges={}
# for i in range(num_merges): #gives dict of all merged pairs : new index values
#     stats=get_stat(ids)
#     pair=max(stats,key=stats.get)
#     idx=256+i
#     # print(f"merging {pair} into {idx}")
#     ids=merge(ids,pair,idx)
#     merges[pair]=idx
# now we have all merged pairs data, stats & merge functions

# vocab={idx:bytes([idx]) for idx in range(256)} # vocabulary for tokenizer
# for (p0,p1),idx in merges.items(): # add marged values into vocab list for decoding
#     vocab[idx]=vocab[p0]+vocab[p1]

class TokenizerV1:
    def __init__(self):
        self.merges={}
        self.vocab={}
    def train(self,text,vocab_size):
        ids = list(text.encode("utf-8"))
        num_merges=vocab_size-256
        for i in range(num_merges):  #gives dict of all merged pairs : new index values
            stats=get_stat(ids)
            pair=max(stats,key=stats.get)
            idx=256+i
            # print(f"merging {pair} into {idx}")
            ids=merge(ids,pair,idx)
            self.merges[pair]=idx
        self.vocab={idx:bytes([idx]) for idx in range(256)}
        for (p0,p1),idx in (self.merges).items(): # add marged values into vocab list for decoding
            self.vocab[idx]=self.vocab[p0]+self.vocab[p1]
    def decode(self,ids):
        decoded=b"".join(self.vocab[idx] for idx in ids)
        return decoded.decode("utf-8",errors="replace")
    def encode(self,text): # we have to convert given text into unicode then merge repeated pairs 
        encoded=list(text.encode("utf-8",errors="replace"))
        while len(encoded)>=2:
            stats=get_stat(encoded)
            pair=min(stats ,key=lambda p: self.merges.get(p,float('inf')))
            if pair not in self.merges:
                break #no pair is repeating
            idx=self.merges[pair]
            encoded=merge(encoded,pair,idx)
        return encoded