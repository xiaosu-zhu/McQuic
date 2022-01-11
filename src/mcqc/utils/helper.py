import os
import torch
import random
import numpy as np
import torch.distributed as dist


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
