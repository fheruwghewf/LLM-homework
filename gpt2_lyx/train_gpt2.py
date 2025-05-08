from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

import torch.optim.adamw

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    
class CasualSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, querry, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not realy a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate querry, key, values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels is the Transformer
        qkv: torch.Tensor = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # attention (materialized the large (T, T) matrix for all the queries and keys)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # flash attention to speed up calculation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head output sie by side
        # output projection
        y = self.c_proj(y)
        return y
    

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits: torch.Tensor = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss: torch.Tensor = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrain(cls, model_type: str):
        '''Loads pretrained GPT-2 model weights from huggingface'''
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretraied gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer = 12, n_head = 12, n_embd = 768),  # 124M params
            'gpt2-medium':  dict(n_layer = 24, n_head = 16, n_embd = 1024), # 350M params
            'gpt2-large':   dict(n_layer = 36, n_head = 20, n_embd = 1280), # 774M params
            'gpt2-xl':      dict(n_layer = 48, n_head = 25, n_embd = 1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        # model_hf = GPT2LMHeadModel.from_pretrained(model_type) 
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2_lyx/gpt2_model/') 

        
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to trasnpose these weigths when import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad) 
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weighted decayed, otherwise no.
        # i.e. all weight tensors in matmults + embeddigs decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [{'params': decay_params, 'weight_decay':weight_decay},
                        {'params': nodecay_params, 'weight_decay': 0.0}]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters')
        print(f'num non-decayed parameters tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters')
        # Create AdamW optimizer and use the fused version if it is avaliable
        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).paramters
        fused_available = True
        use_fused = fused_available and 'cuda' in device
        print(f'using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

                

from tokenizers import Tokenizer, Encoding
tokenizer: Tokenizer = Tokenizer.from_file('gpt2_lyx/gpt2_tokenizer/gpt2_tokenizer')

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in menory
        with open('gpt2_lyx/data/input.txt', 'r') as f:
            text = f.read()

        tokenizer: Tokenizer = Tokenizer.from_file('gpt2_lyx/gpt2_tokenizer/gpt2_tokenizer')

        tokens_encoding: Encoding = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens_encoding.ids)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state 
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bound, reset
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0

        return x, y


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f'using device: ', device)

train_loader = DataLoaderLite(B=4, T=32)

# use tf 32 instead of fp32
torch.set_float32_matmul_precision('high')

# with open('gpt2_lyx/data/input.txt', 'r') as f:
#     text = f.read()

# text = text[:1000]
# tokens_encoding: Encoding = tokenizer.encode(text)
# tokens = tokens_encoding.ids
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])
# buf = buf.to(device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model)
# logits, loss = model(x, y)
# print(loss)

import time

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # linear warmup for warmup_iter steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # if it > lr_decay_iters, return min lr
    if it > max_steps:
        return min_lr
    # in between, use cosine decay down to min lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
    
# optimizer!
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # use bf16 to speedup training on gpu
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    #     logits, loss = model(x, y)
    logits, loss = model(x, y)
    loss: torch.Tensor
    loss.backward()
    # clip the global norm of the gradient at 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # consine decay lr
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / dt
    print(f'step {i:2d}, loss: {loss.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.4f}')


exit(0)


num_return_sequences = 5
max_length = 30

# model = GPT.from_pretrain('gpt2')
model = GPT(GPTConfig())
print(1)
model.eval()
model.to(device)

tokens_encoding = tokenizer.encode("Hello, I'm a launage model,")
tokens = torch.tensor(tokens_encoding.ids, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        print('sampling...', x.size(1) - 8)
        logits = model(x)
        # take the logits at the last position
        logits = logits[:, -1, :]
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface default)
        # topk_probs here become (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)
        # gather the corresponding indicies
        xcol = torch.gather(topk_indicies, -1, ix)
        # append to the sequency
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i].tolist()
    decoded = tokenizer.decode(tokens)
    print('>', decoded)