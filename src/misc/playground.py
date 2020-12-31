from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from mcqc.datasets import SiftLike

torch.backends.cudnn.benchmark = True


def linear(x):
    layer = nn.Linear(x.shape[-1], 4 * x.shape[-1]).cuda()
    optim = torch.optim.Adam(layer.parameters(), 1e-8)
    label = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
    for _ in range(10000):
        optim.zero_grad()
        loss = nn.functional.cross_entropy(layer(x), label)
        loss.backward()
        optim.step()

def conv(x):
    if len(x.shape) == 2:
        x = x[..., None, None]
    layer = nn.Conv2d(x.shape[1], 4 * x.shape[1], (1, 1)).cuda()
    optim = torch.optim.Adam(layer.parameters(), 1e-8)
    label = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
    for _ in range(10000):
        optim.zero_grad()
        loss = nn.functional.cross_entropy(layer(x).squeeze(), label)
        loss.backward()
        optim.step()


def testTime():
    print("start")
    x = torch.randn(256, 1000).cuda()
    tick = time()
    linear(x)
    tock = time()
    print(f"Linear: {tock - tick}")
    tick = time()
    conv(x)
    tock = time()
    print(f"Conv: {tock - tick}")

def testCpu(x):
    loader = DataLoader(x, batch_size=100, num_workers=16, shuffle=True, pin_memory=True)
    i = 0
    for _ in range(10):
        for d in loader:
            d = d.to(f"cuda:{i % 4}")
            d = d ** 2

def testGpu(x):
    loader = DataLoader(x, batch_size=100, shuffle=True, num_workers=0)
    i = 0
    for _ in range(10):
        for d in loader:
            d = d.to(f"cuda:{i % 4}")
            d = d ** 2

def testMoving():
    print("start")
    sift = SiftLike()
    sift.Train(device="cpu")
    tick = time()
    testCpu(sift)
    tock = time()
    print(f"cpu: {tock - tick}")
    sift.Train(device="cuda")
    tick = time()
    testGpu(sift)
    tock = time()
    print(f"gpu: {tock - tick}")

def testSingle(x):
    loader = DataLoader(x, batch_size=1000, shuffle=True, num_workers=0)
    model = nn.Sequential(nn.Linear(128,128), nn.Linear(128, 128)).cuda()
    optim = torch.optim.Adam(model.parameters(), 1e-8)
    label = torch.zeros((1000,), device="cuda", dtype=torch.long)
    for _ in range(100):
        for data in loader:
            optim.zero_grad()
            out = model(data)
            loss = nn.functional.cross_entropy(out, label)
            loss.backward()
            optim.step()


def testMultiple(x):
    loader = DataLoader(x, batch_size=3000, shuffle=True, num_workers=0)
    model = nn.DataParallel(nn.Sequential(nn.Linear(128,128), nn.Linear(128, 128))).cuda()
    optim = torch.optim.Adam(model.parameters(), 1e-8)
    label = torch.zeros((3000,), device="cuda", dtype=torch.long)
    for _ in range(100):
        for data in loader:
            optim.zero_grad()
            out = model(data)
            if out.shape[0] != label.shape[0]:
                continue
            loss = nn.functional.cross_entropy(out, label)
            loss.backward()
            optim.step()

def testDataParallel():
    sift = SiftLike()
    sift.Train()
    tick = time()
    testSingle(sift)
    tock = time()
    print(f"single: {tock - tick}")
    tick = time()
    testMultiple(sift)
    tock = time()
    print(f"multiple: {tock - tick}")

def testEqual():
    ...

if __name__ == "__main__":
    testDataParallel()
