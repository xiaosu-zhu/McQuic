import torch
import numpy as np

from mcqc.datasets.utils import fvecs_read, ivecs_read


def compute_gt():
    baseSet = np.load("data/labelme/train.npy")
    querySet = np.load("data/labelme/query.npy")
    base = torch.from_numpy(baseSet).cuda()
    query = torch.from_numpy(querySet).cuda()
    Sims = list()
    for q in query:
        sim = torch.argmin(((base - q) ** 2).sum(-1))
        Sims.append(sim)
    Sims = torch.stack(Sims, 0)
    np.save("data/labelme/gt.npy", Sims.cpu().numpy())

if __name__ == "__main__":
    compute_gt()
