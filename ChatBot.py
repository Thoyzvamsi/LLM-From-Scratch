import torch
from Tokenization.tokenization import Tokenization
from training.nanogpt import GPTLanguangeModel
import training.nanogpt as nanogpt

"""  Just a simple ChatBot for testing  """

device = nanogpt.device
tokens = Tokenization()

model = GPTLanguangeModel().to(device)
model.load_state_dict(torch.load(nanogpt.weights_path,map_location=device))

question = input("How can I Help You : ")
idx = tokens.encode(question)
idx  = idx.view(1,len(idx))
print(model.generate(idx))