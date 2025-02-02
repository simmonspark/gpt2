# 5️⃣ Weight Decay 적용 그룹 설정

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer

def get_optimizer_grouped_parameters(model, weight_decay=0.01):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        # bias와 LayerNorm 계층 제외
        if "bias" in name or "LayerNorm.weight" in name or 'Embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},  # decay 적용
        {"params": no_decay_params, "weight_decay": 0.0},        # decay 미적용
    ]

# 6️⃣ 옵티마이저 설정
from torch.optim import AdamW

optimizer = AdamW(get_optimizer_grouped_parameters(model), lr=5e-5)

# 7️⃣ TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,  # 기본 weight decay (위에서 설정된 그룹에 따라 적용됨)
    logging_dir="./logs",
)

# 8️⃣ Trainer 실행
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, None),  # 옵티마이저 전달
)

# 9️⃣ 훈련 시작
trainer.train()
