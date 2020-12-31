from scipy.cluster import vq
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import torch
import time
from torch.utils.data import DataLoader

from mcqc.datasets import SiftLike
from mcqc.metrics import Eval, QuantizationError


def OPQTrain(trainSet, M, K):
    x = trainSet

    d = x.shape[-1] // M

    R = np.random.randn(x.shape[-1], x.shape[-1])

    C = np.random.randn(M, K, d).astype(np.float)

    for i in range(M):
        xs = np.split(x, M, -1)
        C[i] = (C[i] * xs[i].std(0)) + xs[i].mean(0)
    for n in range(2): # range(64):
        y = np.zeros_like(x)
        xProj = x @ R
        xs = np.split(xProj, M, -1)


        for i in range(M):
            print("subspace #: %d" % i)
            xSub = xs[i]
            kmeans = MiniBatchKMeans(n_clusters=256, max_iter=64, init=C[i], n_init=1).fit(xSub)

            # [k, d]
            centers = kmeans.cluster_centers_

            C[i] = centers

            # [N, k] =   [N, d] [k, d]
            dist = ((xSub[:, None, :] - C[i]) ** 2).sum(-1)

            # [N, ]
            idx = dist.argmin(axis=1).reshape(-1)

            ySub = centers[idx]
            y[:, i*d:(i+1)*d] = ySub

        print("    opq-np: iter: %d" % n)
        R_opq_np = R

        U, _, Vh = np.linalg.svd(x.T @ y)
        R = U.dot(Vh)


    multiC = np.zeros((M, K, x.shape[-1]))
    for i in range(M):
        multiC[i, :, i*d:(i+1)*d] = C[i]
    return multiC, R_opq_np

def OPQEncode(C, R, dataLoader):
    M, K, D = C.shape
    print(C.shape)
    d = C.shape[-1] // M
    B = list()
    for x in dataLoader:
        x = x.cuda()
        x = x @ R
        xs = torch.split(x, d, -1)
        b = list()
        for i in range(M):
            xSub = xs[i]
            dist = ((xSub[:, None, :] - C[i, :, i*d:(i+1)*d]) ** 2).sum(-1)
            idx = dist.argmin(axis=-1)
            b.append(idx)
        b = torch.stack(b, -1)
        B.append(b)
    B = torch.cat(B, 0)
    return B


if __name__ == "__main__":
    with torch.no_grad():
        sift = SiftLike("SIFT/1M")
        sift.Train(device="cpu")
        C, R = OPQTrain(sift.data.numpy(), 16, 256)
        C = torch.from_numpy(C).cuda()
        R = torch.from_numpy(R).cuda()
        sift.Encode(device="cuda")
        dataLoader = DataLoader(sift, batch_size=10000, shuffle=False, num_workers=0)
        start = time.time()
        B = OPQEncode(C, R, dataLoader)
        end = time.time()
        print(end - start)
        exit()
        print(quantizationError(sift.data @ R, C, B).mean())

        sift.Query(device="cuda")
        sift.data = sift.data @ R.cuda()
        # dataLoader = DataLoader(sift, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        results = Eval.Retrieval(sift.data, C.cuda(), B.cuda())
        sift.Gt()
        recalls = Eval.Recall(results, sift.data[:, :1].cuda()) * 100
        print("R @ 1: %.2f%%" % recalls[0])
        print("R @ 10: %.2f%%" % recalls[9])
        print("R @ 100: %.2f%%" % recalls[99])
