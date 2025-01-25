

import unittest
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from model import GPT


# -----------------------------------------------------------------------------

model_type = 'gpt2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
prompt = "hello gpt? "
model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = 50257
model_config.block_size = 1024
model = GPT(model_config).from_pretrained('gpt2')



model.to(device)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained(model_type)
encoded_input = tokenizer(prompt, return_tensors='pt').to(device)

x1 = encoded_input['input_ids']


logits1, loss = model(x1)


# now draw the argmax samples from each
y1 = model.generate(x1, max_new_tokens=100, do_sample=True)[0]


out1 = tokenizer.decode(y1.cpu().squeeze())

print(out1)