import torch
import numpy as np
import os
import shutil

from mcqc.datasets.utils import fvecsRead, ivecsRead

# trainSet = fvecsRead("data/SIFT/1M/sift_learn.fvecs")
# baseSet = fvecsRead("data/SIFT/1M/sift_base.fvecs")
# querySet = fvecsRead("data/SIFT/1M/sift_query.fvecs")
# gt = ivecsRead("data/SIFT/1M/sift_groundtruth.ivecs")

# assert ((trainSet.astype(int) - trainSet) ** 2).sum() == 0.0
# assert ((baseSet.astype(int) - baseSet) ** 2).sum() == 0.0
# assert ((querySet.astype(int) - querySet) ** 2).sum() == 0.0

# np.save("data/SIFT/1M/train.npy", trainSet.astype(int))
# np.save("data/SIFT/1M/base.npy", baseSet.astype(int))
# np.save("data/SIFT/1M/query.npy", querySet.astype(int))
# np.save("data/SIFT/1M/gt.npy", gt.astype(int))


import h5py

with h5py.File("data/labelme/label.h5", "r") as fp:
    query = fp["query"][()]
    train = fp["train"][()]
    gt = fp["gt"][()]
    np.save("data/labelme/query.npy", query)
    np.save("data/labelme/train.npy", train)
    np.save("data/labelme/gt.npy", gt)
    os.symlink("train.npy", "data/labelme/base.npy")

exit()
