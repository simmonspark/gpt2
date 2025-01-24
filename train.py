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
        self.batch_size = 4
        self.step1_ld = DataLoader(
            self.step1_ds,
            batch_size=self.batch_size,
            num_workers=10,
            prefetch_factor=20,
            shuffle=True
        )
        self.step2_ld = DataLoader(
            self.step2_ds,
            batch_size=self.batch_size,
            num_workers=10,
            prefetch_factor=20,
            shuffle=True
        )

        model_config = GPT.get_default_config()
        model_config.model_type = 'gpt2'
        model_config.vocab_size = 50257
        model_config.block_size = 1024
        self.model = GPT(model_config).from_pretrained('gpt2')

        self.optimizer = self.model.configure_optimizers(model_config)
        self.num_epochs_step1 = 5
        self.num_epochs_step2 = 5
        self.vest_loss = float('inf')
        self.patient = 0
        self.early_stop_patience = 5

    def train_step(self, x, y):
        x = x.to('cuda')
        y = y.to('cuda')
        _, loss = self.model(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train(self):
        if self.is_load:
            self.model.load_state_dict(torch.load('./model.pth'))

        self.model.train()
        self.model.to('cuda')


        for epoch in range(self.num_epochs_step1):
            loss_sum = 0.0
            count = 0
            pbar = tqdm(self.step1_ld, desc=f"Step1 Epoch {epoch+1}/{self.num_epochs_step1}")
            for x, y, pad_mask in pbar:
                loss_val = self.train_step(x, y)
                loss_sum += loss_val
                count += 1
                pbar.set_postfix(loss=loss_val)

            epoch_loss = loss_sum / max(count, 1)
            print(f"[Step1] Epoch {epoch+1}: loss={epoch_loss:.4f}")

            if epoch_loss < self.vest_loss:
                self.vest_loss = epoch_loss
                torch.save(self.model.state_dict(), './model.pth')
                self.patient = 0
            else:
                self.patient += 1

            if self.patient > self.early_stop_patience:
                print("Early stopping in step1 due to no improvement.")
                break

        for epoch in range(self.num_epochs_step2):
            loss_sum = 0.0
            count = 0
            pbar = tqdm(self.step2_ld, desc=f"Step2 Epoch {epoch+1}/{self.num_epochs_step2}")
            for x, y, pad_mask in pbar:
                loss_val = self.train_step(x, y)
                loss_sum += loss_val
                count += 1
                pbar.set_postfix(loss=loss_val)

            epoch_loss = loss_sum / max(count, 1)
            print(f"[Step2] Epoch {epoch+1}: loss={epoch_loss:.4f}")

            if epoch_loss < self.vest_loss:
                self.vest_loss = epoch_loss
                torch.save(self.model.state_dict(), './model.pth')
                self.patient = 0
            else:
                self.patient += 1

            if self.patient > self.early_stop_patience:
                print("Early stopping in step2 due to no improvement.")
                break

if __name__ == '__main__':
    trainer = Trainer(is_load=True)
    trainer.train()
