import torch

def L2DistanceWithNorm(A: torch.Tensor, B: torch.Tensor):
    diff = ((A.unsqueeze(1) - B) ** 2).sum(2)
    maxi, _ = diff.max(1, keepdim=True)
    norm = diff / maxi
    return norm

def CosineSimilarity(A: torch.Tensor, B: torch.Tensor):
    ...

def distanceToSimilarity(distance):
    """Turn a [0, +inf] l2 distance to log-softmaxed similarities: log_softmax([1, 0])

    Args:
        distance ([type]): [description]
    """
    return torch.log_softmax(1 - distance, -1)

def l2distanceToProb(A: torch.Tensor, B: torch.Tensor, temperature: float):
    """Turn a [0, +inf] l2 distance to log-softmaxed similarities: log_softmax([1, 0])

    Args:
        distance ([type]): [description]
    """
    distance = ((A.unsqueeze(1) - B) ** 2).sum(2)
    # l2-norm distance ([0, +inf])
    norm = distance.norm(p=2, dim=1, keepdim=True)
    distance = distance / norm
    distance = distance.max(1, keepdim=True)[0] - distance
    return torch.log_softmax(distance / temperature, -1)
