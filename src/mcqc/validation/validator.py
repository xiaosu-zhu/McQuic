
from typing import Union
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from vlutils.metrics.meter import Meters, Handler
from tqdm import tqdm

from mcqc.datasets.prefetcher import Prefetcher
from mcqc.utils.transforms import DeTransform
from mcqc.validation.meters import MsSSIM, PSNR, BPP
from mcqc.models.compressor import BaseCompressor


class Validator:
    def __init__(self, rank: int):
        self.rank = rank
        self._deTrans = DeTransform().to(rank)
        self._meter = Meters(handlers=[
            MsSSIM().to(rank),
            PSNR().to(rank),
            BPP().to(rank)
        ])

    def tensorToImage(self, x: torch.Tensor) -> torch.Tensor:
        return self._deTrans(x)

    def visualizeIntermediate(self, code: torch.Tensor) -> torch.Tensor:
        code = self._deTrans((code.float() / code.max() - 0.5) * 2)

        n, m, h, w = code.shape

        code = code.reshape(n * m, 1, h, w)[:32]
        code = F.interpolate(code, scale_factor=4, mode="nearest")
        return code

    @torch.inference_mode()
    def count(self, epoch: int, model: BaseCompressor, trainLoader: Union[DataLoader, Prefetcher]):
        for image in tqdm(trainLoader, dynamic_ncols=True, bar_format="Est [%3d] {n_fmt}/{total_fmt} |{bar}|" % epoch, leave=False, disable=self.rank != 0):
            model.count(image.to(self.rank, non_blocking=True))

    @torch.inference_mode()
    def validate(self, epoch: int, model: BaseCompressor, valLoader: DataLoader):
        with model._quantizer.readyForCoding() as cdfs:
            for images in tqdm(valLoader, dynamic_ncols=True, bar_format="Val [%3d] {n_fmt}/{total_fmt} |{bar}|" % epoch, leave=False, disable=self.rank != 0):
                images = images.to(self.rank, non_blocking=True)
                codes, binaries, header = model.compress(images, cdfs)
                restored = model.decompress(codes, binaries, cdfs, header)
                self._meter(images=self.tensorToImage(images), binaries=binaries, restored=self.tensorToImage(restored))
        return self._meter.results(), self._meter.summary()

    def test(self, testLoader: DataLoader): raise NotImplementedError
