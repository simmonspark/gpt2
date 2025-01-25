import torch
from transformers import GPT2LMHeadModel
from torchinfo import summary
from model import GPT

model = GPT2LMHeadModel.from_pretrained('gpt2')
input_size = (1, 1024)
summary(model, input_size=input_size, dtypes=[torch.long],depth=20)
print(model.state_dict().keys())


