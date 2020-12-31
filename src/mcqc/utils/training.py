import torch
import math

def SearchNewBatchSize(baseBatchSize: int, totalAmount: int):
    for i in range(baseBatchSize, 0, -1):
        if totalAmount % i == 0:
            return i
    raise RuntimeError("Can't find appropriate batch size, use a larger batch and retry")


def SetNewLr(newBatchSize: int, oldBatchSize: int, *optimizers: torch.optim.Optimizer):
    # Follow the sqrt() lr adjusting strategy
    scale = math.sqrt(newBatchSize / oldBatchSize)
    for optim in optimizers:
        for g in optim.param_groups:
            g["lr"] = g["lr"] * scale
