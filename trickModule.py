import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

class RMSNorm(torch.nn.Module): # a variant of LayerNorm that is faster and more memory efficient
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # set the weight to be 1 initially
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_pos_emb(dim: int, end: int, theta: float = 10000.0):
    # Encoding Word Oder In Complex Embeddings https://arxiv.org/abs/1912.12333
    # ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING https://arxiv.org/pdf/2104.09864.pdf
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) 
    # freqs is a 1d decreasing tensor, with length of dim/2
    # max = 1, min = 1/theta^(0.99999)
    # desending is fast at the beginning, then slow down
    t = torch.arange(end, device=freqs.device)  # t is the positional index, from 0 to end
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_emb = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # convert the lenght 1 and angle array to complex number coordinates
    # even dim is real, which = cos(angle); odd dim is imaginary, which = sin(angle)
    # larger the dim, slower the value oscillation
    return pos_emb

def reshape_for_broadcast(pos_emb: torch.Tensor, x: torch.Tensor):
    '''
    this function is used to reshape the pos_emb tensor to the same shape as the input x
    '''
    ndim = x.ndim
    #assert pos_emb.shape == (x.shape[1], x.shape[-1]), f"pos_emb.shape = {pos_emb.shape}, x.shape = {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return pos_emb.view(*shape)


def apply_pos_emb(
    xq: torch.Tensor, # xq.shape = (batch_size, seqlen, n_head, n_embd // n_head)
    xk: torch.Tensor, # xk.shape = (batch_size, seqlen, n_head, n_embd // n_head)
    pos_emb: torch.Tensor, # pos_emb.shape = (seqlen, n_embd // n_head // 2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    # example xq.shape = (1, 10, 2, 12)
    # reshape to (1, 10, 2, 6, 2), because view_as_complex() requires the last dim to be 2
    # after view_as_complex(), shape = (1, 10, 2, 6)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    pos_emb = reshape_for_broadcast(pos_emb, xq_)

    # current xq_.shape = (1, 10, 2, 6, 2)
    # after xq_ * pos_emb, shape = (1, 10, 2, 6)
    # after view_as_real(), shape = (1, 10, 2, 6, 2)
    # after flatten(3), shape = (1, 10, 2, 12)
    xq_out = torch.view_as_real(xq_ * pos_emb).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_emb).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)