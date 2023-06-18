import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from trickModule import RMSNorm, precompute_pos_emb, apply_pos_emb


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # query, key, values projections for all heads
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # output projection
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        
    def forward(self, x: Tensor, pos_emb: Tensor, start_idx: int):
        batch_size, seqlen, emb_dim = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # split embedding dim into number of heads
        xq = xq.view(batch_size, seqlen, self.n_head, emb_dim // self.n_head)
        xk = xk.view(batch_size, seqlen, self.n_head, emb_dim // self.n_head)
        xv = xv.view(batch_size, seqlen, self.n_head, emb_dim // self.n_head)

        # apply positional embeddings
        xq, xk = apply_pos_emb(xq, xk, pos_emb[start_idx:start_idx+seqlen])

        # TODO: add cache for fast inference

        # transpose to get dimensions batch_size * n_head * seqlen * emb_dim
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seqlen, emb_dim) # merge heads

        # output projection
        return self.wo(y)



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.fc2  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)  # using in LLAMA
        x = self.fc2(x)
        x = self.dropout(x)
        return x



class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, pos_emb: Tensor, start_idx: int):
        x = x + self.attn(self.ln_1(x), pos_emb, start_idx=start_idx)
        x = x + self.mlp(self.ln_2(x))
        return x




class GPTConfig:
    max_token_len: int = 4096
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0



class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_token_len is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            w_token_emb = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=-1),
            drop = nn.Dropout(config.dropout),
            layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_final = RMSNorm(config.n_embd), # final layer normalization
        ))
        self.emb_to_logits = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.w_token_emb.weight = self.emb_to_logits.weight # https://paperswithcode.com/method/weight-tying

        # initialize weights
        self.apply(self._init_weights)

        # initialize positional embeddings
        self.pos_emb = precompute_pos_emb(config.n_embd // config.n_head, config.max_token_len * 2)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.00, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.00, std=0.01)


    def forward(self, tokens, start_idx=0, training=True): # return all logits
        batch_size, seqlen = tokens.size()

        # create token embeddings from token table; x.shape = (batch_size, seqlen, n_embd)
        h = self.transformer.w_token_emb(tokens)
        assert h.size() == (batch_size, seqlen, self.config.n_embd)

        self.pos_emb = self.pos_emb.to(h.device)
        
        # transformer layers
        for layer in self.transformer.layers:
            h = layer(h, self.pos_emb[start_idx:start_idx+seqlen], start_idx=start_idx)

        logits = self.emb_to_logits(self.transformer.ln_final(h))

        return logits.float()


    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        #n_params = sum(p.numel() for p in self.parameters())
        n_params = sum(p.numel() for p in self.transformer.layers[0].parameters() if p.requires_grad)
        return n_params

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
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
