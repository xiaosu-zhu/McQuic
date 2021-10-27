from typing import Any
import os
import shutil

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from absl import app, flags
from cfmUtils.config import read
import torchvision
from torchvision import transforms

from mcqc.models.compressor import PQCompressorBig
from mcqc.datasets import Basic
from mcqc.utils.vision import DeTransform, getTestTransform
from mcqc import Config

FLAGS = flags.FLAGS

flags.DEFINE_string("cfg", "", "The config.json path.")
flags.DEFINE_string("device", "cuda", "The device to use.")
flags.DEFINE_string("ckpt", "", "The checkpoint path.")
flags.DEFINE_string("dataset", "data/clic/valid", "The images path")

class EntropyEstimator:
    def __init__(self, config: Config, model: PQCompressorBig, dataset: Dataset, device: str):
        self._model = model
        self._device = device
        self._model.to(device)
        self._dataset = dataset
        self._dataLoader = DataLoader(dataset, pin_memory=True)
        self._deTrans = DeTransform()
        self._config = config

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        baseDir = "tmp/visualize"
        shutil.rmtree(baseDir, ignore_errors=True)
        os.makedirs(baseDir, exist_ok=True)
        codes = [[[] for _ in range(self._config.Model.m)] for _ in range(self._model._levels)]
        for i, x in enumerate(self._dataLoader):
            x = x.to(self._device, non_blocking=True)
            allZs, allHards, allCodes = self._model.getLatents(x)
            for j, (z, quantized, b) in enumerate(zip(allZs, allHards, allCodes)):
                # [m, h, w]
                for m, mb in enumerate(b[0]):
                    codes[j][m].append(mb.flatten())

        bitsPerToken = []
        for l in range(self._model._levels):
            bits = []
            for m in range(self._config.Model.m):
                code = torch.cat(codes[l][m])
                count = torch.bincount(code.flatten(), minlength=self._config.Model.k[l])
                prob = count / count.sum()
                estimateEntropy = prob.log2()
                estimateEntropy[estimateEntropy == float("-inf")] = 0
                estimateEntropy = -(prob * estimateEntropy)
                bits.append(estimateEntropy)
            bitsPerToken.append(bits)
        return bitsPerToken

class Visualizer:
    def __init__(self, config: Config, model: PQCompressorBig, dataset: Dataset, device: str):
        self._model = model
        self._device = device
        self._model.to(device)
        self._dataset = dataset
        self._dataLoader = DataLoader(dataset, pin_memory=True)
        self._deTrans = DeTransform()
        self._config = config

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        bitsPerToken = args[0]

        baseDir = "tmp/visualize"
        shutil.rmtree(baseDir, ignore_errors=True)
        os.makedirs(baseDir, exist_ok=True)
        bTrans = DeTransform(0.0, 1.0)
        for i, x in enumerate(tqdm(self._dataLoader)):
            os.makedirs(os.path.join(baseDir, str(i)))
            nowDir = os.path.join(baseDir, str(i))
            torchvision.io.write_png(self._deTrans(x).cpu()[0], os.path.join(nowDir, "raw.png"))

            x = x.to(self._device, non_blocking=True)


            allZs, allHards, allCodes = self._model.getLatents(x)

            _, _, H, W = x.shape

            requiredBits = []
            for j, (z, quantized, b) in enumerate(zip(allZs, allHards, allCodes)):
                b = b.cpu()
                k = self._config.Model.k[j]
                # [0~1]
                bNormed = b / float(k)
                bImg = bTrans(bNormed)
                # [c // m, h, w], [c // m, h, w]
                for m, (zs, qs) in enumerate(zip(z[0], quantized[0])):
                    # [1, h, w]
                    zs = zs.mean(0, keepdim=True)
                    qs = qs.mean(0, keepdim=True)
                    zs = DeTransform(zs.min(), zs.max())(zs)
                    qs = DeTransform(qs.min(), qs.max())(qs)

                    rs = zs - qs
                    rs = DeTransform(rs.min(), rs.max())(rs)

                    zs = F.interpolate(zs[None, ...], scale_factor=16, mode="nearest")
                    qs = F.interpolate(qs[None, ...], scale_factor=16, mode="nearest")
                    rs = F.interpolate(rs[None, ...], scale_factor=16, mode="nearest")
                    torchvision.io.write_png(zs[0].cpu(), os.path.join(nowDir, f"z-level{j}-group{m}.png"))
                    torchvision.io.write_png(qs[0].cpu(), os.path.join(nowDir, f"q-level{j}-group{m}.png"))
                    torchvision.io.write_png(rs[0].cpu(), os.path.join(nowDir, f"r-level{j}-group{m}.png"))


                # [m, h, w]
                for m, (mb, bi) in enumerate(zip(bImg[0], b[0])):
                    mb = mb[None, None, ...].expand(1, 3, mb.shape[0], mb.shape[1])
                    mb = F.interpolate(mb, scale_factor=16, mode="nearest")
                    torchvision.io.write_png(mb[0], os.path.join(nowDir, f"b-level{j}-group{m}.png"))

                    # [k] entropies
                    bits = bitsPerToken[j][m]
                    # [h, w] required bits
                    requiredBitsJM = bits[bi]
                    h, w = requiredBitsJM.shape
                    actualBits = requiredBitsJM * h * w / float(H * W)
                    scale = H / h
                    requiredBits.append(F.interpolate(actualBits[None, None, ...], scale_factor=scale, mode="nearest"))
                    # RGBA
                    requiredBitsJM = requiredBitsJM[None, None, ...].expand(1, 4, requiredBitsJM.shape[0], requiredBitsJM.shape[1])
                    requiredBitsJM = DeTransform(requiredBitsJM.min(), requiredBitsJM.max())(requiredBitsJM)
                    requiredBitsJM = F.interpolate(requiredBitsJM, scale_factor=16, mode="nearest")
                    # Red is always red
                    requiredBitsJM[0, 0] = 255
                    # G and B set to zero
                    requiredBitsJM[0, 1:3] = 0
                    requiredBitsJM = transforms.ToPILImage()(requiredBitsJM[0])
                    requiredBitsJM.save(os.path.join(nowDir, f"requiredBits-level{j}-group{m}.png"))
                    # torchvision.io.write_png(requiredBits[0].cpu(), os.path.join(nowDir, f"requiredBits-level{j}-group{m}.png"))
            requiredBits = sum(requiredBits)
            # RGBA
            requiredBits = requiredBits.expand(1, 4, requiredBits.shape[-2], requiredBits.shape[-1])
            requiredBits = DeTransform(requiredBits.min(), requiredBits.max())(requiredBits)
            # Red is always red
            requiredBits[0, 0] = 255
            # G and B set to zero
            requiredBits[0, 1:3] = 0
            requiredBits = transforms.ToPILImage()(requiredBits[0])
            requiredBits.save(os.path.join(nowDir, f"requiredBits-full.png"))


class Plotter:
    def __init__(self, config: Config, model: PQCompressorBig, dataset: Dataset, device: str):
        self._model = model
        self._device = device
        self._model.to(device)
        self._dataset = dataset
        self._dataLoader = DataLoader(dataset, pin_memory=True)
        self._deTrans = DeTransform()
        self._config = config

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        baseDir = "tmp/tsne"
        shutil.rmtree(baseDir, ignore_errors=True)
        os.makedirs(baseDir, exist_ok=True)

        totalZs = [[[] for _ in range(self._config.Model.m)] for _ in range(self._model._levels)]
        totalHards = [[[] for _ in range(self._config.Model.m)] for _ in range(self._model._levels)]
        totalCodes = [[[] for _ in range(self._config.Model.m)] for _ in range(self._model._levels)]

        for i, x in enumerate(tqdm(self._dataLoader)):
            x = x.to(self._device, non_blocking=True)

            allZs, allHards, allCodes = self._model.getLatents(x)

            for j, (z, quantized, b) in enumerate(zip(allZs, allHards, allCodes)):
                # [h, w], [c // m, h, w], [c // m, h, w]
                for m, (bs, zs, qs) in enumerate(zip(b[0], z[0], quantized[0])):
                    # [k, c]
                    # codebook = self._model._quantizers[j][m]._codebook

                    # [h*w, c]
                    zs = zs.reshape(zs.shape[0], -1).permute(1, 0)
                    qs = qs.reshape(qs.shape[0], -1).permute(1, 0)

                    totalZs[j][m].append(zs)
                    totalHards[j][m].append(qs)
                    totalCodes[j][m].append(bs.flatten())

        for j in trange(self._model._levels):
            for m in trange(self._config.Model.m, leave=False):
                # [N, c]
                z = torch.cat(totalZs[j][m])
                # [N, c]
                q = torch.cat(totalHards[j][m])
                # [N]
                b = torch.cat(totalCodes[j][m])
                # [K]
                counts = torch.bincount(b)
                # [10] indices
                topK = torch.argsort(counts, descending=True)[:20]

                # [N, 10] sum -> [N] > 0 -> at least one hit
                included = (b[:, None] == topK).sum(-1) > 0

                b = b[included]
                z = z[included]
                q = q[included]

                tqdm.write(str(b.shape))

                uniqueCodes, ivf = torch.unique(b, sorted=True, return_inverse=True)
                codebook = self._model._quantizers[j][m]._codebook
                centers = codebook[uniqueCodes]

                vectorsToVis = z
                # vectorsToVis = torch.cat((z, centers))
                result = TSNE(perplexity=45, min_grad_norm=1e-12, learning_rate="auto", init="pca", n_jobs=-1).fit_transform(vectorsToVis.cpu().numpy())

                # categories = torch.cat([ivf.reshape(-1), torch.arange(len(pickedCodes))]).cpu().tolist()
                categories = [(ivf / float(len(uniqueCodes))).cpu().tolist(),
                                (torch.arange(len(uniqueCodes)) / float(len(uniqueCodes))).cpu().tolist()]
                # categories = (torch.arange(len(pickedCodes)) / float(len(pickedCodes))).tolist()
                resultA = result[:len(z)]
                # resultB = result[len(z):]
                plt.scatter(resultA[:, 0], resultA[:, 1], c=categories[0], s=2)
                # plt.scatter(resultB[:, 0], resultB[:, 1], c=categories[1], marker="*", s=100)
                plt.savefig(os.path.join(baseDir, f"level-{j}-group-{m}.pdf"))
                plt.close()



@torch.inference_mode()
def main(_):
    with logging_redirect_tqdm():
        config = read(FLAGS.cfg, None, Config)
        dataset = Basic(FLAGS.dataset, transform=getTestTransform())
        model = PQCompressorBig(config.Model.m, config.Model.k, config.Model.channel, config.Model.withGroup, config.Model.withAtt, False, config.model.alias, -1)
        # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        savedModel = torch.load(FLAGS.ckpt, map_location={"cuda:0": "cpu"})
        model.load_state_dict({k[len("module._compressor."):]: v for k, v in savedModel["model"].items() if "module._compressor." in k})

        model = model.eval()

        bitsPerToken = EntropyEstimator(config, model, dataset, FLAGS.device)()

        visualizer = Visualizer(config, model, dataset, FLAGS.device)
        visualizer(bitsPerToken)

        plotter = Plotter(config, model, Basic("data/clic/test", transform=getTestTransform()), FLAGS.device)
        plotter()



if __name__ == "__main__":
    app.run(main)
