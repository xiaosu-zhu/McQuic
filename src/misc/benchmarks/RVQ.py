from scipy.cluster import vq
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import torch
import time
from torch.utils.data import DataLoader

from mcqc.datasets import SiftLike
from mcqc.metrics import Eval, QuantizationError


def reconstruct(C, B):
    M = C.shape[0]
    x = np.zeros((B.shape[0], C.shape[-1]))
    for i in range(M):
        x += C[i][B[:, i]]
    return x

def RVQTrain(trainSet, M, K):
    x = trainSet.copy()
    D = x.shape[-1]

    C = np.random.randn(M, K, D)

    C = (C * x.std(0)) + x.mean(0)

    B = np.zeros((x.shape[0], M), np.int)


    for i in range(1): # range(32):
        residual = x
        for m in range(M):
            print(f"stage {m}")
            kmeans = MiniBatchKMeans(n_clusters=256, init=C[m], n_init=1).fit(residual)
            # [k, d]
            codebook = kmeans.cluster_centers_

            C[m] = codebook

            # [N, k] =   [N, d] [k, d]
            dist = ((residual[:, None, :] - codebook) ** 2).sum(-1)

            # [N, ]
            idx = dist.argmin(axis=1).reshape(-1)

            quantized = codebook[idx]
            B[:, m] = idx
            residual = residual - quantized
        print(f"iter {i}")
    for i in range(0):
        print(f"refine iter {i}")
        randomM = np.random.permutation(M)
        for m in randomM:
            print(f"refine on {m}")
            idx = list(range(M))
            idx.pop(m)
            remainCodebook = C[idx]
            remainCodes = B[:, idx]
            residual = trainSet - reconstruct(remainCodebook, remainCodes)
            kmeans = MiniBatchKMeans(n_clusters=256, init=C[m], n_init=1).fit(residual)
            # [k, d]
            codebook = kmeans.cluster_centers_
            C[m] = kmeans.cluster_centers_
            # [N, k] =   [N, d] [k, d]
            dist = ((residual[:, None, :] - codebook) ** 2).sum(-1)

            # [N, ]
            idx = dist.argmin(axis=1).reshape(-1)
            B[:, m] = idx
    return C

def RVQEncode(C, dataLoader):
    M, K, D = C.shape
    B = list()
    for x in dataLoader:
        residual = x.cuda()
        b = list()
        for i in range(M):
            dist = ((residual[:, None, :] - C[i]) ** 2).sum(-1)
            idx = dist.argmin(axis=-1)
            b.append(idx)

            quantized = C[i][idx]
            residual = residual - quantized
        b = torch.stack(b, -1)
        B.append(b)
    B = torch.cat(B, 0)
    return B


if __name__ == "__main__":
    with torch.no_grad():
        sift = SiftLike("SIFT/1M")
        sift.Train(device="cpu")
        C = RVQTrain(sift.data.numpy(), 16, 256)
        C = torch.from_numpy(C).cuda()
        sift.Encode(device="cuda")
        dataLoader = DataLoader(sift, batch_size=10000, shuffle=False, num_workers=0)
        start = time.time()
        B = RVQEncode(C, dataLoader)
        end = time.time()
        print(end - start)
        exit()
        print(quantizationError(sift.data, C, B).mean())

        sift.Query(device="cuda")
        # dataLoader = DataLoader(sift, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        results = Eval.Retrieval(sift.data, C.cuda(), B.cuda())
        sift.Gt()
        recalls = Eval.Recall(results, sift.data[:, :1].cuda()) * 100
        print("R @ 1: %.2f%%" % recalls[0])
        print("R @ 10: %.2f%%" % recalls[9])
        print("R @ 100: %.2f%%" % recalls[99])
