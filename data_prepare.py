from torch.utils.data import Dataset
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer


def prepare_data1():
    data_path = '/media/sien/media/data/RMlabel.json'
    tmp = []
    with open(data_path, 'r') as f:
        data = json.load(f)
        for i in tqdm(data['data_info']):
            question = i['question']
            answer = i['answer01']['contents']
            tmp.append({'question': question, 'answer': answer})
    with open('./general_QA', 'w', encoding='utf-8') as f:
        json.dump(tmp, f, ensure_ascii=False, indent=4)


class general_dataset(Dataset):
    def __init__(self):
        # mode = train or rlhf
        super(general_dataset, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        with open('./general_QA', 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answer = self.data[idx]['answer']
        ret = self.tokenizer(question + ' ' + answer, padding='max_length', truncation=True, max_length=1024,
                             return_tensors='pt')
        input_ids = ret['input_ids'].squeeze(0)
        pad_mask = ret['attention_mask'].squeeze(0)

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.tokenizer.eos_token_id
        return input_ids.int(), labels, pad_mask


class RLHF_dataset(Dataset):
    def __init__(self):
        super(RLHF_dataset, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        with open('/media/sien/media/data/RLHF_200.json', 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answer = self.data[idx]['answer']
        data = question + ' ' + answer
        ret = self.tokenizer(data, padding='max_length', truncation=True, max_length=1024, return_tensors='pt')
        input_ids = ret['input_ids'].squeeze(0)
        pad_mask = ret['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.tokenizer.eos_token_id
        return input_ids.int(), labels, pad_mask
if __name__ == '__main__':
    prepare_data1()
