import torch
from transformers import GPT2LMHeadModel
from torchinfo import summary

model = GPT2LMHeadModel.from_pretrained('gpt2')
input_size = (1, 512)
summary(model, input_size=input_size, dtypes=[torch.long],depth=20)