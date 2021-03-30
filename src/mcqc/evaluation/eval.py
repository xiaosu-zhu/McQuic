from logging import Logger
from typing import Tuple
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import center_crop
from tqdm import tqdm
import numpy as np
from cfmUtils.saver import Saver
from cfmUtils.runtime import Timer

from mcqc import Consts
from cfmUtils.datasets import Zip

from .helpers import _EVALSSIM, psnr


class Eval:
    def __init__(self, savePath: str, dataset: Dataset, model, device: str = "cuda", logger: Logger = None):
        self._device = device
        self._model = model.to(device)
        Saver.load(savePath, logger, model=self._model)
        self._savePath = savePath
        self._model.eval()
        self._dataset = dataset
        torch.autograd.set_grad_enabled(False)
        self._logger = logger or Consts.Logger

    def _encode(self):
        dataLoader = DataLoader(self._dataset, batch_size=1, shuffle=False, num_workers=4)
        B = list()
        ticker = Timer()
        for x in tqdm(dataLoader, ncols=40, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            x = x.to(self._device, non_blocking=True)
            latents = self._model._encoder(x)
            codes = self._model._quantizer.encode(latents)
            B.append(codes.detach().cpu())
        interval, _ = ticker.tick()
        self._logger.info("Encode %d samples for %.2f seconds, %.2e s/sample, %.2e samples/s", len(self._dataset), interval, interval / len(self._dataset), len(self._dataset) / interval)
        B = torch.cat(B, 0).numpy()
        return B

    def _decode(self, b: torch.Tensor, size):
        quantized = self._model._quantizer.quantize(b)
        restored = self._model._decoder(quantized)
        return center_crop(restored, size)

    def test(self):
        B = self._encode()
        saveDir = os.path.dirname(self._savePath)
        np.save(os.path.join(saveDir, "B.npy"), B.cpu().numpy())
        self._logger.info("Save B.npy at %s", saveDir)
        testLoader = DataLoader(Zip(self._dataset, B), batch_size=1, shuffle=False, num_workers=4)
        ssims = list()
        psnrs = list()
        for raw, b in testLoader:
            restored = self._decode(b, raw.shape[-2:])
            ssims.append(_EVALSSIM(restored, raw))
            psnrs.append(psnr(restored, raw))
        self._logger.info("MS-SSIM: %.4e", sum(ssims) / len(ssims))
        self._logger.info("   PSNR: %.4e", sum(psnrs) / len(psnrs))
