from scipy.cluster import vq
import numpy as np
from mcqc.datasets import SiftLike
from mcqc import Eval
import torch

from time import time


def Run():
    train = SiftLike()
    train.Train(device="cpu")
    x = train.data.numpy()
    a = time()
    codebook, _ = vq.kmeans(x, 256, 1)
    b = time()
    print(b - a)
    train.Encode(device="cpu")
    x = train.data.numpy()
    a = time()
    code, _ = vq.vq(x, codebook)
    b = time()
    print(b - a)
    quantized = codebook[code]
    print(((x - quantized) ** 2).sum(-1).mean())
    tester = Eval(True, None, None, None)
    tester.Test(torch.from_numpy(codebook[None, ...]).cuda(), torch.from_numpy(code).long().cuda())

if __name__ == "__main__":
    Run()
