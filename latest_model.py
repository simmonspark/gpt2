import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

'''
GPT2LMHeadModel                                    [1, 12, 1024, 64]         --
├─GPT2Model: 1-1                                   [1, 12, 1024, 64]         --
│    └─Embedding: 2-1                              [1, 1024, 768]            38,597,376
│    └─Embedding: 2-2                              [1, 1024, 768]            786,432
│    └─Dropout: 2-3                                [1, 1024, 768]            --
│    └─ModuleList: 2-4                             --                        --
│    │    └─GPT2Block: 3-1                         [1, 1024, 768]            --
│    │    │    └─LayerNorm: 4-1                    [1, 1024, 768]            1,536
│    │    │    └─GPT2SdpaAttention: 4-2            [1, 1024, 768]            --
│    │    │    │    └─Conv1D: 5-1-> need T         [1, 1024, 2304]           1,771,776
│    │    │    │    └─Conv1D: 5-2-> need T         [1, 1024, 768]            590,592
│    │    │    │    └─Dropout: 5-3                 [1, 1024, 768]            --
│    │    │    └─LayerNorm: 4-3                    [1, 1024, 768]            1,536
│    │    │    └─GPT2MLP: 4-4                      [1, 1024, 768]            --
│    │    │    │    └─Conv1D: 5-4                  [1, 1024, 3072]           2,362,368
│    │    │    │    └─NewGELUActivation: 5-5       [1, 1024, 3072]           --
│    │    │    │    └─Conv1D: 5-6-> need T         [1, 1024, 768]            2,360,064
│    │    │    │    └─Dropout: 5-7                 [1, 1024, 768]            --
========================================================================================
GPT2Config {
  "_name_or_path": "gpt2",
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_inner": null,
  "n_layer": 12,
  "n_positions": 1024,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
'''


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))


# attention을 진행하면서 dropout 2번 들어감
class MySelfAttention(nn.Module):
    def __init__(self):
        super(MySelfAttention, self).__init__()
        self.c_attn = nn.Linear(768, 2304, bias=True)
        self.c_proj = nn.Linear(768, 768, bias=True)
        self.register_buffer('attention_mask', torch.tril(torch.ones((1, 1, 1024, 1024))))
        self.attention_drop1 = nn.Dropout(0.1)
        self.attention_drop2 = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(768, dim=2)
        k = k.view(B, T, 12, C // 12).transpose(1, 2)
        q = q.view(B, T, 12, C // 12).transpose(1, 2)
        v = v.view(B, T, 12, C // 12).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.attention_mask[:, :, :T, :T] == 0, float('-inf'))
        if mask is not None:
            mask = mask.view(B, 1, T, 1)
            mask = mask.expand(B, 1, T, T)  # batch,head,seq,seq for attention energy
            att = att.masked_fill(mask == 0, 1e-4)
        att = F.softmax(att, dim=-1)
        att = self.attention_drop1(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.attention_drop2(self.c_proj(y))
        return y


class GPT2MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.c_fc = nn.Linear(768, 768 * 4)
        self.c_proj = nn.Linear(768 * 4, 768)
        self.act = NewGELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)

        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(768, )
        self.attn = MySelfAttention()
        self.ln_2 = nn.LayerNorm(768)
        self.mlp = GPT2MLP()

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        '''C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.weight_decay = 0.1
        C.learning_rate = 5e-4
        C.betas = (0.9, 0.999)
        return C'''
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(50257, 768),
            wpe=nn.Embedding(1024, 768),
            drop=nn.Dropout(0.1),
            h=nn.ModuleList([Block() for _ in range(12)]),
            ln_f=nn.LayerNorm(768)
        ))
        self.lm_head = nn.Linear(768, 50257, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * 12))

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def forward(self, x, y=None, mask=None):
        device = 'cuda'
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # 위치 임베딩
        tok_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, None)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # 확률분포
        loss = None
        if y is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, pad_mask=None, temperature=0.8, do_sample=False, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
             "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=5e-4,
            betas=[0.9, 0.999]
        )
        return optimizer

    def load_hf_weight(self):
        # Hugging Face GPT2 모델 로드
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        print(model_hf.config)

        # Hugging Face state_dict 가져오기
        sd_hf = model_hf.state_dict()
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')]  # 필터링된 키
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # 사용자 정의 GPT 모델 초기화
        config = {
            'model_type': 'gpt2',
            'vocab_size': 50257,
            'block_size': 1024
        }
        model = GPT()  # 사용자 정의 GPT 모델 생성
        sd = model.state_dict()  # 사용자 정의 모델의 state_dict

        # 키 일관성 검증
        assert len(keys) == len([k for k in sd if not k.endswith('attn.bias')])

        # 가중치 복사
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())  # 전치(transpose) 처리된 가중치 복사
            else:
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])  # 일반 가중치 복사

        # state_dict를 사용자 정의 모델에 로드
        model.load_state_dict(sd)
        return model


if __name__ == '__main__':
    '''tmp = torch.randint(1,100,size = (4,1024),device = 'cuda')
    model = GPT().to('cuda')
    logit = model.generate(tmp,100,None)'''
    model = GPT().load_hf_weight()
    # batch per 3~4G vram
    from transformers import GPT2Tokenizer
    #mode= model.load_state_dict(torch.load('model.pth',weights_only=True))
    prompt = "Hello!!!!!!!!!!!!!!!!, my name is"
    model.to('cuda')
    model.eval()


    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    encoded_input = tokenizer(prompt, return_tensors='pt').to('cuda')

    x1 = encoded_input['input_ids']

    logits1, loss = model(x1)

    # now draw the argmax samples from each
    y1 = model.generate(x1, max_new_tokens=100, do_sample=False)[0]

    out1 = tokenizer.decode(y1.cpu().squeeze())

    print(out1)
