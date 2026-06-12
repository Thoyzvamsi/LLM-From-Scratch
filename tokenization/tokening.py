import torch

class Tokenization:
    def __init__(self,text):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(set(text))

    # -- Encoding to Tokens --
    def encoding(self,context_letters):
        encoded_numbers = []
        for letter in context_letters:
            num = self.chars.index(letter)
            tokens.append(num)
        tokens = torch.tensor(tokens)
        return tokens
    
    # -- Decoding to text -- 
    def decoding(self,context_numbers):
        decoded_text = ""
        for number in context_numbers:
            encoded_text = "".join(self.chars[number])
        return encoded_text