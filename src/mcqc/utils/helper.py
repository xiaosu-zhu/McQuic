import os

from rich.progress import BarColumn, Progress, SpinnerColumn, TimeElapsedColumn
import torch
from torch import nn
import torch.distributed as dist
import random
import numpy as np

def initializeProcessGroup(port: str, rank: int, worldSize: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    torch.autograd.set_detect_anomaly(False) # type: ignore
    torch.backends.cudnn.benchmark = True # type: ignore
    torch.manual_seed(3407)
    random.seed(3407)
    torch.cuda.set_device(rank)
    np.random.seed(3407)
    dist.init_process_group("nccl", world_size=worldSize, rank=rank)


def getRichProgress():
    return Progress("Epoch [{task.fields[epoch]:%4d}]: {task.completed:4d}/{task.total:4d}", SpinnerColumn(), BarColumn(), TimeElapsedColumn(), "D = {task.fields[loss]}", transient=True, disable=dist.get_rank() != 0)


class EMATracker(nn.Module):
    def __init__(self, size: torch.Size, momentum: float = 0.9):
        super().__init__()
        self._momentum = momentum
        self.register_buffer("_shadow", torch.empty(size) * torch.nan)

    def forward(self, x: torch.Tensor):
        if torch.all(torch.isnan(self._shadow)):
            return self._shadow.copy_(x)
        self._shadow -= self._momentum * (self._shadow - x)
        return self._shadow
