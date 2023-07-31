"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RetNetRelPos(nn.Module):
    def __init__(self, embed_dim=16, heads_num=2, recurrent_chunk_size=4):
        super().__init__()
        self.heads_num = heads_num
        self.embed_dim = embed_dim
        self.embed_dim_per_head = embed_dim // heads_num
        self.recurrent_chunk_size = recurrent_chunk_size
        
        # 'Angle':
        # 
        # Shape: [embed_dim_per_head]
        
        # Values range from 1 to 1/10,000 via an exponential decay
        # and looks like [1, 1, 0.5, 0.5, 0.25, 0.25...] such that
        # each element repeats and has a constant ratio to the next
        # two elements
        angle = 1.0 / (10000 ** torch.linspace(0, 1, self.embed_dim_per_head // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        
        
        # 'Decay':
        #
        # Shape: [num_heads]
        
        # Values are such that each element is negative the
        # magnitude exponentially decays by 1 / 2
        decay = torch.log(1 - 2 ** (-5 - torch.arange(heads_num, dtype=torch.float)))

        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            cross_decay = cross_decay[:, None, None]
            inner_decay = inner_decay[:, :, None] / (scale / scale[:, -1, None])
            retention_rel_pos = ((sin, cos), (mask, cross_decay, inner_decay))
        else:
            # Make a range from 0 to slen with the same
            # device and dtype as self.decay
            index = torch.arange(slen).to(self.decay)
    
            # Because index is [length] and angle is [key_dim]
            # multiplyiny a [length, 1] and a [1, key_dim]
            # makes us a matrix of [length, per_head_dim],
            # which means it's suitable for multiplying by
            trig_base = index[:, None] * self.angle[None, :]
            # Per-row increment up every row by a constant amount,
            # which makes sense because this is over time unit
            
            # Per-column values increment DOWN every 2 columns
            # by a constant exponential decay. 
            
            # So for a 6-length x 4-key_dim matrix we could
            #
            #tensor([[0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
            #        [1.0e+00, 1.0e+00, 1.00-04, 1.0e-04],
            #        [2.0e+00, 2.0e+00, 2.0e-04, 2.0e-04],
            #        [3.0e+00, 3.0e+00, 3.0e-04, 3.0e-04],
            #        [4.0e+00, 4.0e+00, 4.0e-04, 4.0e-04],
            #        [5.0e+00, 5.0e+00, 5.0e-04, 5.0e-04]])
            sin = torch.sin(trig_base)
            cos = torch.cos(trig_base)
            
            # torch.tril returns a lower-triangular part of a matrix,
            # with everything above the diagonal set to 0
            # So for a length of 4 this is like
            # [[1, 0, 0, 0]
            #  [1, 1, 0, 0]
            #  [1, 1, 1, 0]
            #  [1, 1, 1, 1]]
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            # [[1, inf, inf, inf]
            #  [2,   1, inf, inf]
            #  [3,   2,   1, inf]
            #  [4,   3,   2,   1]]
            
            # Mask is of [len, len], so we can multiply
            # by [head_length, 1, , 1]
            # to get a per-head mask like [head, len, len]
            mask = torch.exp(mask * self.decay[:, None, None])
        
            # For overflow?
            mask = torch.nan_to_num(mask)
            
            # Normalize per-row, but NOT by the sum-per-row
            # so that each row sums to 1, but by the square
            # root of the sum per row
            # WTF
            # WTF need explanation for this
            mask_norm = mask.sum(dim=-1, keepdim=True).sqrt()
            mask = mask / mask_norm
                  
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# rotate_every_two
#
# Interchange each two elements along the last
# dimension of the input, and get the negativ
# of the first
#
# print(rotate_every_two(torch.eye(4,4).view(1, 1, 4, 4)))
# tensor([[[[-0.,  1., -0.,  0.],
#           [-1.,  0., -0.,  0.],
#           [-0.,  0., -0.,  1.],
#           [-0.,  0., -1.,  0.]]]])
def rotate_every_two(x):

    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    # Concatenate along a new dimension, which we
    # will shortly forget
    x = torch.stack((-x2, x1), dim=-1)
    # Return something with the same dimension
    # as the original X, but with every other element
    # along the last dimension interchanged and with
    # a negative created
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError

class Retention(nn.Module):
    def __init__(
        self,
        embed_dim,
        heads_num,
        value_factor=2,
        gate_fn="swish",
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.heads_num = heads_num
        # The value transforms are scaled up over
        # the size of the key / query transforms
        # by value_factor
        self.value_factor = value_factor
        
        self.kq_dim = embed_dim // heads_num
        self.v_dim = embed_dim * value_factor // heads_num
        
        # Classic pre-softmax scaling of the heads to avoid
        # gradient saturation; in all transformers
        self.scaling = self.kq_dim ** -0.5

        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * value_factor, bias=True)
        self.g_proj = nn.Linear(embed_dim, embed_dim * value_factor, bias=True)

        self.out_proj = nn.Linear(embed_dim * value_factor, embed_dim, bias=True)

        self.group_norm = nn.LayerNorm(self.v_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
    

    def parallel_forward(self, qr, kr, v, mask):
        bs, leng, _ = v.size()

        vr = v.view(bs, leng, self.heads_num, self.v_dim).transpose(1, 2)
        # [bs, heads_num, length, v_dim]

        qk_mat = qr @ kr.transpose(-1, -2) # bs * heads_num * leng * leng
        
        # mask is heads_num * leng * leng, so all good
        qk_mat = qk_mat * mask
        
        # what is this half-normalization thing
        
        qk_mat = qk_mat / qk_mat.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output

    def recurrent_forward(self, qr, kr, v, decay, incremental_state):
        
        bs, _, __ = v.shape

        v = v.view(bs, self.heads_num, self.v_dim, 1)
        
        # kr: [bs, leng = 1, heads_num, kq_dim]
        kv = kr * v
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            kv = prev_kv * (1 - 1 / scale).view(self.num_heads, 1, 1) + kv / scale.view(self.num_heads, 1, 1)
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        else:
            scale = torch.ones_like(decay)

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale

        output = torch.sum(qr * kv, dim=3)
        return output
    
    def forward(
        self,
        x,
        rel_pos,
        chunkwise_recurrent=False,
        incremental_state=None
    ):
        
        heads_num = self.heads_num
        embed_dim = self.embed_dim
        kq_dim = self.kq_dim # embed_dim // heads_num
        bs, leng, _ = x.size()
        
        # Note that the rel_pos is the same
        # per layer, although if you do a recurrent mode
        # it changes whether it is first instance of recurrence
        # or not
        # sin: (length, key_dim)
        # cos: (length, key_dim)
        # mask: (heads, length, length)
        (sin, cos), inner_mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bs, leng, heads_num, self.kq_dim).transpose(1, 2)
        k = k.view(bs, leng, heads_num, self.kq_dim).transpose(1, 2)

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
            raise NotImplemented("Not implemented")
        elif chunkwise_recurrent:
            #output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
            raise NotImplemented("Not implemented")
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask)

        output = self.group_norm(output).reshape(bs, leng, heads_num * self.v_dim)

        output = self.gate_fn(g) * output

        output = self.out_proj(output)

        return output

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        #self.attn = CausalSelfAttention(config)
        self.attn = Retention(
            embed_dim=config.n_embd,
            heads_num=config.n_head
        )
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, rel_pos):
        x = x + self.attn(self.ln_1(x), rel_pos)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            rel_pos = RetNetRelPos(config.n_embd, config.n_head, recurrent_chunk_size=16),
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, incremental_states=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        rel_pos = self.transformer.rel_pos(t)
        x = self.transformer.drop(tok_emb + pos_emb)

        if incremental_states is not None:
            for i in range(len(self.transformer.h)):
                if i not in incremental_states:
                    incremental_states[i] = {}
        
        if incremental_states is not None:
            for i, block in enumerate(self.transformer.h):
                x = block(x, rel_pos, incremental_states[i])
        else:
            for block in self.transformer.h:
                x = block(x, rel_pos)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        incremental_states = {}
        for _ in range(max_new_tokens):
            idx_cond = idx
            logits, _ = self(idx_cond, None, incremental_states)
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
