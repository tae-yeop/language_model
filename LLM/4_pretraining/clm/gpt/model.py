import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect

from config import GPTConfig

class LayerNorm(nn.Module):
    """
    LayerNorm with an optional bias
    """
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.bias = nn.Parameter(torch.zeros(cfg.hidden_size)) if cfg.bias else None
        
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.eps = 1e-6

    def forward(self, x):
        irms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x*irms*self.weight
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.hidden_size, 3* cfg.hidden_size, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.c_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size)
            )

        self.n_heads = cfg.n_heads
        self.hidden_size = cfg.hidden_size

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)

        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        if not self.flash:
            attn_weights = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
            mask = self.mask[:, :, :T, :T] == 0
            att = attn_weights.masked_fill(mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, 
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=True
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))

        return y


class RotaryEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, cfg.hidden_size, 2).float() / cfg.hidden_size))
        self.register_buffer("inv_freq", self.inv_freq)

    def get_emb(self, seq_len):
        t = torch.arange(seq_len)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.hidden_size, 4 * cfg.hidden_size, bias=cfg.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.hidden_size, cfg.hidden_size, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(cfg.vocab_size, cfg.hidden_size),
                wpe = nn.Embedding(cfg.block_size, cfg.hidden_size),
                drop = nn.Dropout(cfg.dropout),
                h = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layer)]),
                ln_f = LayerNorm(cfg),
            )
        )
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * cfg.n_layer))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        # embedding dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        # final layer norm
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def crop_block_size(self, block_size):
        assert block_size <= self.cfg.block_size
        self.cfg.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'mask'):
                block.attn.mask = block.attn.mask[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(self, model_type, override_args=None):
        override_args = override_args or {}
        assert all(k=='dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        
        config_args = {
            'gpt2' : dict(n_layer=12, n_heads=12, hidden_size=768),
            'gpt2-medium':  dict(n_layer=24, n_heads=16, hidden_size=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_heads=20, hidden_size=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_heads=25, hidden_size=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['mask'] = True # always True for GPT model checkpoints

        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']
        
        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.mask')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

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

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type=='cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        계산 처리량 (MFU, Model Flops Utilization)
        한 토큰당 FLOPs (Floating Point Operations): PaLM 논문의 Appendix B를 참조하여, 모델이 하나의 토큰을 처리하는 데 필요한 FLOPs를 계산
        공식 : 6×N+12×L×H×Q×T
        N (모델의 총 파라미터 수)
        L×H×Q×T (어텐션 메커니즘 관련)
        """
        N = self.get_num_params()
        cfg = self.cfg
        L, H, Q, T = cfg.n_layer, cfg.n_heads, cfg.hidden_size//cfg.n_heads, cfg.block_size
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
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / tempartrue # [B, vocab_size]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx



