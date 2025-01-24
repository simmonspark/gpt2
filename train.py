import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
from data_prepare import general_dataset, RLHF_dataset
from model import GPT

class Trainer():
    def __init__(self, is_load, state_path=None):
        self.is_load = is_load
        self.state_path = state_path
        self.step1_ds = general_dataset()
        self.step2_ds = RLHF_dataset()
        self.step1_ld = DataLoader(self.step1_ds, batch_size=6, num_workers=10, prefetch_factor=20,
                                   sampler=torch.utils.data.RandomSampler(self.step1_ds, replacement=True,
                                                                          num_samples=10000))
        self.step2_ld = DataLoader(self.step2_ds, batch_size=6,  num_workers=10, prefetch_factor=20,
                                   sampler=torch.utils.data.RandomSampler(self.step2_ds, replacement=True,
                                                                          num_samples=1000))
        self.model = GPT(is_pretrain=True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
        self.epoch = 100
        self.vest_loss = float('inf')
        self.patient = 0

    def train(self):
        if self.is_load:
            self.model.load_state_dict(torch.load(self.state_path))
        self.model.train()
        self.model.to('cuda')
        pbar = tqdm(range(self.epoch))
        for i in pbar:
            loss_tmp = []
            for x, y, pad_mask in self.step1_ld:
                x = x.to('cuda')
                y = y.to('cuda')
                pad_mask = pad_mask.to('cuda')
                _, loss = self.model(x, y, pad_mask=None)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_tmp.append(loss.item())
            epo_loss = sum(loss_tmp) / len(loss_tmp)
            if epo_loss < self.vest_loss:
                self.vest_loss = epo_loss
                torch.save(self.model.state_dict(), './model.pth')
                self.patient = 0
            else:
                self.patient += 1
            if self.patient > 5:
                print("Early stopping due to no improvement.")
                break

            pbar.set_postfix(loss=epo_loss)
            print(f'{i} : epo loss: {epo_loss:.4f}')
if __name__ == '__main__':
    trainer = Trainer(is_load=False)
    trainer.train()