import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class NewGELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super(CausalSelfAttention, self).__init__()
        self.c_attn = nn.Linear(768, 3 * 768)
        self.c_proj = nn.Linear(768, 768)
        self.drop = nn.Dropout(0.1)
        self.res_drop = nn.Dropout(0.1)
        self.register_buffer('logical_mask', torch.tril(torch.ones(1024).unsqueeze(0)).view(1, 1, 1024, 1))

    def forward(self, tu):
        x = tu[0]
        pad_mask = tu[1]
        B, N, C = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        k = k.view(B, N, 12, C // 12).transpose(1, 2)
        q = q.view(B, N, 12, C // 12).transpose(1, 2)
        v = v.view(B, N, 12, C // 12).transpose(1, 2)
        att = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(k.size(-1)))
        att.masked_fill(self.logical_mask == 0, float('-inf'))

        if pad_mask is not None:
            pad_mask = pad_mask.view(B,1,N,1)
            att = att.masked_fill(pad_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = att @ v
        att = att.transpose(1, 2).contiguous().view(B, N, C)
        att = self.res_drop(self.c_proj(att))
        return att


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(768)
        self.ln_2 = nn.LayerNorm(768)
        self.attn = CausalSelfAttention()
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(768, 4 * 768),
            c_proj=nn.Linear(4 * 768, 768),
            act=NewGELU(),
            dropout=nn.Dropout(0.1),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        idx = x[0]
        pad_mask = x[1]

        idx = idx + self.attn((self.ln_1(idx), pad_mask))
        idx = idx + self.mlpf(self.ln_2(idx))
        return idx

class GPT(nn.Module):
    def __init__(self, is_pretrain=False):
        super(GPT, self).__init__()

        self.pre = is_pretrain

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(50257, 768),
            wpe=nn.Embedding(1024, 768),
            drop=nn.Dropout(0.1),
            h=nn.ModuleList([Block() for _ in range(12)]),
            ln_f=nn.LayerNorm(768),
        ))
        self.lm_head = nn.Linear(768, 50257, bias=False)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * 12))
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

        if self.pre:
            self.load_weights()

    def load_weights(self):
        gpt_origin = GPT2LMHeadModel.from_pretrained('gpt2')
        state = gpt_origin.state_dict()
        keys = [k for k in state if not k.endswith('attn.masked_bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        model = GPT()
        ret_state = model.state_dict()

        for k in keys:
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    ret_state[k].copy_(state[k].t())
            else:
                with torch.no_grad():
                    ret_state[k].copy_(state[k])
        return model.load_state_dict(ret_state)


    def forward(self, x,y = None, pad_mask = None):
        device = x.device
        pos = torch.arange(0, x.shape[1], dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block((x,pad_mask))
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, max_length, temperature=1, do_sample=False, top_k=None, endoftext_id=50256):
        """
        Generate tokens with max_length padding.

        Parameters:
        - idx: Tensor of token indices (input sequence).
        - max_new_tokens: Number of new tokens to generate.
        - max_length: Fixed length for padding.
        - temperature: Sampling temperature.
        - do_sample: Whether to sample or take argmax.
        - top_k: If set, limits sampling to top-k tokens.
        - endoftext_id: Token ID indicating the end of text.
        """
        # Ensure input is padded to max_length
        pad_token_id = tokenizer.pad_token_id
        assert pad_token_id is not None, "Tokenizer must have a pad_token_id defined."

        # Pad input idx to max_length
        if idx.size(1) < max_length:
            padding = torch.full((idx.size(0), max_length - idx.size(1)), pad_token_id, dtype=idx.dtype,
                                 device=idx.device)
            idx = torch.cat((idx, padding), dim=1)

        # Track where new tokens start
        start_idx = idx.size(1) - max_length

        for _ in range(max_new_tokens):
            # Compute logits
            logits, _ = self(idx)

            # Only use logits for the last non-padding token
            logits = logits[:, start_idx - 1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Compute probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample or take argmax
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # Add the next token to the sequence
            idx[:, start_idx] = idx_next[:, 0]

            # Increment start_idx for the next token
            start_idx += 1

            # Stop if end-of-text token is generated
            if endoftext_id is not None and (idx_next == endoftext_id).any():
                break


        # Ensure output remains padded to max_length
        if idx.size(1) < max_length:
            padding = torch.full((idx.size(0), max_length - idx.size(1)), pad_token_id, dtype=idx.dtype,
                                 device=idx.device)
            idx = torch.cat((idx, padding), dim=1)

        return idx


if __name__ == '__main__':
    model = GPT(is_pretrain=True)

    # 입력 문장
    input_text = "hello gpt."
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # 토큰화
    enc = tokenizer(
        input_text,
        max_length=1024,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = enc['input_ids']

    # 생성
    output_ids = model.generate(input_ids, max_new_tokens=10, max_length=1024, temperature=0.8, do_sample=True)

    # 디코딩
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Generated text:", output_text)