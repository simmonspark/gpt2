import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
from data_prepare import general_dataset, RLHF_dataset
from model import GPT
from new_imple import GPT2, get_pretrained_model


class Trainer():
    def __init__(self, is_load, state_path=None):
        self.is_load = is_load
        self.state_path = state_path

        self.step1_ds = general_dataset()
        self.step2_ds = RLHF_dataset()
        self.step1_ld = DataLoader(
            self.step1_ds, batch_size=4, num_workers=10, prefetch_factor=20,
            sampler=torch.utils.data.RandomSampler(self.step1_ds, replacement=True, num_samples=5000)
        )
        self.step2_ld = DataLoader(
            self.step2_ds, batch_size=4, num_workers=10, prefetch_factor=20,
            sampler=torch.utils.data.RandomSampler(self.step2_ds, replacement=True, num_samples=1000)
        )

        model_config = GPT.get_default_config()
        model_config.model_type = 'gpt2'
        model_config.vocab_size = 50257
        model_config.block_size = 1024
        #self.model = GPT(model_config).from_pretrained('gpt2')
        self.model = get_pretrained_model()
        '''from latest_model import GPT as G
        self.model = G().load_hf_weight()
        '''
        self.optimizer = self.model.configure_optimizers(model_config)
        self.max_iters = 5000
        self.vest_loss = float('inf')
        self.patient = 0

    def train(self):
        if self.is_load:
            self.model.load_state_dict(torch.load('./model.pth', weights_only=True))

        self.model.train()
        self.model.to('cuda')
        data_iter = iter(self.step1_ld)

        pbar = tqdm(range(self.max_iters))
        for i in pbar:

            try:
                x, y, pad_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(self.step1_ld)
                x, y, pad_mask = next(data_iter)

            x = x.to('cuda')
            y = y.to('cuda')
            pad_mask = pad_mask.to('cuda')

            _, loss = self.model(x, y, pad_mask)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            loss_val = loss.item()
            print(loss_val)
            pbar.set_postfix(loss=loss_val)

            if loss_val < self.vest_loss:
                self.vest_loss = loss_val
                torch.save(self.model.state_dict(), './model.pth')
                self.patient = 0
            else:
                self.patient += 1

            '''if self.patient > 500:
                print("Early stopping due to no improvement.")
                break'''



if __name__ == '__main__':

    import multiprocessing

    multiprocessing.set_start_method('spawn')
    trainer = Trainer(is_load=True)
    trainer.train()
