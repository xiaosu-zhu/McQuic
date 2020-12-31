import torch
from torch import nn
from torch.utils.data import DataLoader
import time

from mcqc.metrics import QuantizationError, Eval
from mcqc.datasets import SiftLike
from mcqc import Consts

def Train(model: nn.Module, optim: torch.optim.Optimizer, scheduler, dataLoader: DataLoader):
    for i in range(100):
        for data in dataLoader:
            optim.zero_grad()
            _, loss = model(data)
            loss.backward()
            optim.step()
        print(f"Epoch {i} with Loss: {loss.detach().mean()}")
        scheduler.step()

def Encode(model: nn.Module, dataLoader: DataLoader):
    codes = list()
    for data in dataLoader:
        code, _ = model(data)
        codes.append(code)
    codes = torch.cat(codes, 0)
    codebook = model.Codebook()
    return codebook, codes



def Run(model: nn.Module, lr=1.):
    sift = SiftLike("labelme").Train()
    sift.data *= 100.0
    # N, D = sift.shape
    dataLoader = DataLoader(sift, batch_size=1000, shuffle=True, num_workers=0)
    optim = torch.optim.Adam(model.parameters(), lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.99)
    Train(model, optim, scheduler, dataLoader)
    del dataLoader
    with torch.no_grad():
        sift.Encode("cuda")
        sift.data *= 100.0
        dataLoader = DataLoader(sift, batch_size=1000, shuffle=False, num_workers=0)
        model.eval()
        start = time.time()
        codebook, codes = Encode(model, dataLoader)
        end = time.time()
        print(end - start)
        print(quantizationError(sift.data, codebook, codes).mean())
        del dataLoader
        sift.Query(device="cuda")
        sift.data *= 100.0
        # dataLoader = DataLoader(sift, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        results = Eval.Retrieval(sift.data, codebook.cuda(), codes.cuda())
        sift.Gt()
        recalls = Eval.Recall(results, sift.data[:, :1].cuda()) * 100
        print("R @ 1: %.2f%%" % recalls[0])
        print("R @ 10: %.2f%%" % recalls[9])
        print("R @ 100: %.2f%%" % recalls[99])
