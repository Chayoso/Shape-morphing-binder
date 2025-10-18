"""Gumbel-Softmax sampling utilities."""

import torch
import torch.nn.functional as F
from typing import Optional
from ..utils.config import CLAMP_GUMBEL


def generate_gumbel_noise(batch_M: int, N: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """Generate Gumbel noise for sampling."""
    u = torch.rand(batch_M, N, generator=generator, device=device)
    u = torch.clamp(u, *CLAMP_GUMBEL)
    g = -torch.log(-torch.log(u))
    return g


def gumbel_softmax_onehot(
    probs: torch.Tensor,
    M: int,
    tau: float = 0.2,
    hard: bool = True,
    seed: int = 0,
    generator: Optional[torch.Generator] = None,
    batch_size: int = 5000
) -> torch.Tensor:
    """Draw M samples from categorical distribution using Gumbel-Softmax."""
    N = probs.shape[0]
    device = probs.device
    
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(seed)
    
    Y_list = []
    num_batches = (M + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, M)
        batch_M = end_idx - start_idx
        
        g = generate_gumbel_noise(batch_M, N, generator, device)
        
        safe_probs = torch.clamp(probs, min=1e-10)
        logits = (safe_probs.log().unsqueeze(0) + g) / max(tau, 1e-6)
        y_soft = F.softmax(logits, dim=1)
        
        if hard:
            idx = y_soft.argmax(dim=1)
            y_hard = F.one_hot(idx, num_classes=N).float()
            y_batch = y_hard - y_soft.detach() + y_soft
        else:
            y_batch = y_soft
        
        Y_list.append(y_batch)
        
        del g, logits, y_soft
        if hard:
            del idx, y_hard
    
    Y = torch.cat(Y_list, dim=0)
    return Y