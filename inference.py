

import unittest
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from model import GPT


# -----------------------------------------------------------------------------

model_type = 'gpt2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
prompt = "동물 생명에 관해 어떻게 생각해? "

model_config = GPT()

model_hf = GPT2LMHeadModel.from_pretrained(model_type)  # init a HF model too
model = GPT()
model.load_state_dict(torch.load("./model.pth", map_location=device))
# ship both to device
model.to(device)
model_hf.to(device)

# set both to eval mode
model.eval()
model_hf.eval()

# tokenize input prompt
# ... with mingpt
tokenizer = GPT2Tokenizer.from_pretrained(model_type)
encoded_input = tokenizer(prompt, return_tensors='pt').to(device)

x1 = encoded_input['input_ids']


logits1, loss = model(x1)


# now draw the argmax samples from each
y1 = model.generate(x1, max_new_tokens=1024, do_sample=True)[0]


out1 = tokenizer.decode(y1.cpu().squeeze())

print(out1)