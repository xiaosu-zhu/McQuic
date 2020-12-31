import torch

def quantizationError(X: torch.Tensor, C: torch.Tensor, B: torch.Tensor):
    N, M = B.shape
    # [batch, M]
    ix = torch.arange(M).expand_as(B)
    # [batch, M, D]
    gatheredCodewords = C[[ix, B]]
    return ((X - gatheredCodewords.sum(1)) ** 2).sum(-1)

def reconstruct(C, B):
    N, M = B.shape
    # [batch, M]
    ix = torch.arange(M).expand_as(B)
    # [batch, M, D]
    gatheredCodewords = C[[ix, B]]
    return gatheredCodewords.sum(1)
