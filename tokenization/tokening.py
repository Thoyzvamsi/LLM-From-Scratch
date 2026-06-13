import torch

class Tokenization:
    def __init__(self,text):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(set(text))

    # -- Encoding to Tokens --
    def encoding(self,context_letters):
        tokens = []
        for letter in context_letters:
            num = self.chars.index(letter)
            tokens.append(num)
        tokens = torch.tensor(tokens)
        return tokens
    
    # -- Decoding to text -- 
    def decoding(self,context_numbers):
        decoded_text = ""
        for number in context_numbers:
            decoded_text = decoded_text+self.chars[number]
        return decoded_text