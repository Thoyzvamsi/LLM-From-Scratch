import pickle

class Traintokeinzer:
    def __init__(self):
        self.text = open("data\data.txt",'r',encoding="utf-8").read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = 500
        self.merges={}
        self.vocab={}

    def merge(self,ids,pair,idx): #merges most repeated pairs into single index values
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
    
    def get_stat(self,ids): #gives dictionary of token pair:number of times repeated
        count={}
        for pair in zip(ids,ids[1:]):
            count[pair]=count.get(pair,0)+1
        return count
    
    def train(self):
        ids = list(self.text.encode("utf-8"))
        num_merges = self.vocab_size - 256

        for i in range(num_merges):  #gives dict of all merged pairs : new index values
            stats = self.get_stat(ids)
            pair = max(stats,key=stats.get)
            idx = 256+i
            print(f"merging {pair} into {idx}")
            ids = self.merge(ids,pair,idx)
            self.merges[pair]=idx
        self.vocab = {idx:bytes([idx]) for idx in range(256)}

        for (p0,p1),idx in (self.merges).items(): # add marged values into vocab list for decoding
            self.vocab[idx]=self.vocab[p0]+self.vocab[p1]

if __name__ == "__main__":
    tok = Traintokeinzer()
    tok.train()

    with open(r"data\tokenizer.pkl", "wb") as f:
        pickle.dump(
            {
                "vocab_size": tok.vocab_size,
                "merges": tok.merges,
                "vocab": tok.vocab,
            },
            f,
        )

    print("Tokenizer saved!")


'''   
steps to run :
1. Run: python -m Tokenization.train_tokenizer

This does:

train()
  ↓
creates merges
  ↓
creates vocab
  ↓
saves tokenizer.pkl
2. then we can use Tokenization

'''
 