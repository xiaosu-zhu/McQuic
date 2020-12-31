import multiprocessing as mp
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from mcqc.solvers import LSolver, ISolver, SSolver
from mcqc.metrics import QuantizationError, Eval
from mcqc.datasets import SiftLike, Zip

class LSQ(nn.Module):
    def __init__(self, M, K, D):
        super().__init__()
        self._m = M
        self._k = K
        self._d = D
        self.register_buffer("codebook", torch.randn(M * K, D))

        self.register_buffer("_shift", torch.arange(self._m) * self._k)
        self.register_buffer("_cUnary", (self.codebook ** 2).sum(-1)[None, ...])
        self.register_buffer("_cPair", 2 * (self.codebook @ self.codebook.t()))

    def Codebook(self):
        return self.codebook.reshape(self._m, self._k, self._d)


    def Update(self, codebook):
        self.codebook.copy_(codebook.reshape(self._m*self._k, self._d))
        self._cUnary.copy_((self.codebook ** 2).sum(-1)[None, ...])
        self._cPair.copy_(2 * (self.codebook @ self.codebook.t()))


    def forward(self, x, b, npert, icmIter):
        N, M = b.shape
        oldB = b.clone()
        x_ci = -2 * (x @ self.codebook.t())

        b += self._shift

        if npert > 0:
            uniform = torch.ones_like(b, dtype=float) / M
            # [N, npert]
            pertidx = torch.multinomial(uniform, npert)
            # [N, npert], where each row = [i, i, i..., i]
            ix = torch.arange(N)[:, None].expand_as(pertidx)
            pertvals = torch.randint(self._k, (N, npert), device=b.device) + pertidx * self._k
            # [N, npert]
            b[[ix, pertidx]] = pertvals

        mIdx = torch.randperm(M)
        for _ in range(icmIter):
            for i in mIdx:
                otherBs = b[:, i != mIdx]
                b[:, i] = torch.argmin(x_ci[:, i*self._k:(i + 1)*self._k] + self._cUnary[:, i*self._k:(i+1)*self._k] + self._cPair[i*self._k:(i+1)*self._k, otherBs].sum(2).t(), 1) + self._k * i

        b -= self._shift

        oldQE = quantizationError(x, self.Codebook(), oldB)
        newQE = quantizationError(x, self.Codebook(), b)
        worse = newQE >= oldQE
        b[worse] = oldB[worse]
        return b

def Train(model, sift, nILS, nICM, nPERT):
    solver = ISolver(model._m, model._k)
    model.Update((torch.randn(model._m, model._k, model._d, device="cuda") * sift.data.std(0)) + sift.data.mean(0))
    B = torch.randint(model._k, (sift.data.shape[0], model._m), device=sift.data.device)
    for _ in range(1):
        for _ in range(nILS):
            B = model(sift.data, B, nPERT, nICM)
        model.Update(solver.solve(sift.data, B))
        print(quantizationError(sift.data, model.Codebook(), B).mean())

def Encode(model, sift, nILS, nICM, nPERT):
    solver = ISolver(model._m, model._k)
    B = torch.randint(model._k, (sift.data.shape[0], model._m), device=sift.data.device)
    # for _ in range(nILS):
    #     B = model(sift.data, B, nPERT, nICM)
    dataLoader = DataLoader(Zip(sift.data, B), batch_size=100000, shuffle=False, num_workers=0)
    B = list()
    for x, b in dataLoader:
        for _ in range(nILS):
            b = model(x, b, nPERT, nICM)
        B.append(b)
    B = torch.cat(B, 0)
    return solver.solve(sift.data, B), B # model.Codebook(), B

@torch.no_grad()
def Run(model: nn.Module, nILS, nICM, nPERT, encodeILS):
    sift = SiftLike("SIFT/1M").Train()
    # N, D = sift.shape
    Train(model, sift, nILS=nILS, nICM=nICM, nPERT=nPERT)
    sift.Encode("cuda")
    model.eval()
    start = time.time()
    codebook, codes = Encode(model, sift, nILS=encodeILS, nICM=nICM, nPERT=nPERT)
    end = time.time()
    print(end - start)
    print(quantizationError(sift.data, codebook, codes).mean())
    sift.Query(device="cuda")
    # dataLoader = DataLoader(sift, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    results = Eval.Retrieval(sift.data, codebook.cuda(), codes.cuda())
    sift.Gt()
    recalls = Eval.Recall(results, sift.data[:, :1].cuda()) * 100
    print("R @ 1: %.2f%%" % recalls[0])
    print("R @ 10: %.2f%%" % recalls[9])
    print("R @ 100: %.2f%%" % recalls[99])


if __name__ == "__main__":
    lsq = LSQ(16, 256, 128).cuda()
    Run(lsq, 8, 4, 2, 32)
