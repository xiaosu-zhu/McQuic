from typing import Type, Callable, Iterator
from logging import Logger

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from cfmUtils.saver import Saver
from cfmUtils.vision.colorSpace import rgb2hsv, hsv2rgb
from pytorch_msssim import ms_ssim

from mcqc.algorithms.algorithm import Algorithm
from mcqc.evaluation.helpers import evalSSIM, psnr
from mcqc.models.whole import Whole
from mcqc import Consts, Config

def _ssimExp(source, target, datarange):
    return (2.7182818284590452353602874713527 - ms_ssim(source, target, data_range=datarange).exp()) / (1.7182818284590452353602874713527)


class Plain(Algorithm):
    def __init__(self, config: Config, model: Whole, device: str, optimizer: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer], scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler], saver: Saver, continueTrain: bool, logger: Logger):
        super().__init__()
        self._model = model
        if device == "cuda" and torch.cuda.device_count() > 1:
            self._model = nn.DataParallel(self._model.to(device))
        else:
            self._model = self._model.to(device)
        self._device = device
        # self._optimizerD = optimizer(1e-5, self._model.module._discriminator.parameters(), 0)
        self._optimizer = optimizer(config.LearningRate, self._model.parameters(), 0)
        # self._schedulerD = scheduler(self._optimizerD)
        self._scheduler = scheduler(self._optimizer)
        self._saver = saver
        self._logger = logger
        self._config = config
        self._continue = continueTrain

    @staticmethod
    def _deTrans(imaage):
        return ((imaage * 0.5 + 0.5) * 255).clamp(0.0, 255.0).byte()

    def run(self, trainLoader: torch.utils.data.DataLoader, testLoader: torch.utils.data.DataLoader):
        initTemp = 1.0
        minTemp = 0.1
        step = 0
        flag = False
        count = 0
        regCoeff = self._config.Coef.reg
        dB = 0.0
        target = 21.0
        cv = 0.1
        maxCV = 0.1

        if self._continue:
            loaded = self._saver.load(self._saver.SavePath, self._logger, model=self._model)# , optimG=self._optimizerG, schdrG=self._schedulerG, step=step, temp=initTemp)
            # initTemp = loaded["temp"]
            # step = loaded["step"]

        dB = self._eval(testLoader, step, flag)

        for i in range(self._config.Epoch):
            self._model.train()
            for images in trainLoader:
                images = images.to(self._device, non_blocking=True)
                (ssimLoss, l1l2Loss, reg), (restored, codes, latents, logits, quantizeds) = self._model(images, flag, cv)
                self._optimizer.zero_grad()
                (self._config.Coef.ssim * ssimLoss + self._config.Coef.l1l2 * l1l2Loss + self._config.Coef.reg * reg).mean().backward()
                # torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=10.0)
                self._optimizer.step()
                # self._saver.add_scalar("loss/gLoss", gLoss.mean(), global_step=step)
                self._saver.add_scalar("loss/ssimLoss", ssimLoss.mean(), global_step=step)
                self._saver.add_scalar("loss/l1l2Loss", l1l2Loss.mean(), global_step=step)
                self._saver.add_scalar("loss/reg", reg.mean(), global_step=step)
                if (step + 1) % 100 == 0:
                    self._saver.add_images("train/raw", self._deTrans(images), global_step=step)
                    self._saver.add_images("train/res", self._deTrans(restored), global_step=step)
                    self._saver.add_histogram("code", codes[0].reshape(-1), bins=256, max_bins=256, global_step=step)
                if (step + 1) % 1000 == 0:
                    dB = self._eval(testLoader, step, flag)
                    if dB > target and not flag:
                        flag = True
                        self._logger.info("Insert Transformer", int(target))
                    self._saver.save(self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step+1, temp=initTemp)
                    self._logger.info("%3dk steps complete, update: LR = %.2e, T = %.2e, count = %d", (step + 1) // 1000, self._scheduler.get_last_lr()[0], initTemp, count)
                if (step + 1) % 10000 == 0 and 100000 < step < 130000:
                    # self._schedulerD.step()
                    self._scheduler.step()
                    self._logger.info("reduce lr")
                # initTemp = max(initTemp * 0.9999, minTemp)
                step += 1
                # cv *= min(cv * 1.0001, maxCV)
                # mixin *= 0.9999

    @torch.no_grad()
    def _reInitializeCodebook(self, dataLoader, c):
        self._model.eval()
        if isinstance(self._model, nn.DataParallel):
                model = self._model.module._compressor
        else:
            model = self._model._compressor
        model = model.cuda()
        bs = list()
        for raw in dataLoader:
            raw = raw.to(self._device, non_blocking=True)
            # restored, _, _, _, _ = self._model(raw, 0.5, True, 0.0)
            latents = model._encoder(raw)
            b = model._quantizer.encode(latents)
            bs.append(b[0].detach().cpu())
        # [n, h, w]
        bs = torch.cat(bs, 0).numpy()
        unique, count = np.unique(bs, return_counts=True)
        total = bs.size
        # print(c.shape)
        c = c.t()
        remain = c.shape[0] - len(unique)
        current = 0
        notUsedC = c[list(x for x in range(c.shape[0]) if x not in unique)]
        std = c[unique].std()
        for u, thisCount in zip(unique, count):
            proportion = thisCount / total
            thisPiece = int(proportion * remain)
            codeword = c[u]
            reinitCodewords = notUsedC[current:current+thisPiece]
            current += thisPiece
            reinitCodewords.data.copy_(torch.from_numpy(np.random.randn(thisPiece, c.shape[-1]) * (float(std) ** 2) + codeword.detach().cpu().numpy()))
        c[list(x for x in range(c.shape[0]) if x not in unique)] = notUsedC
        return c.t()

    @torch.no_grad()
    def _eval(self, dataLoader: torch.utils.data.DataLoader, step: int, transform: bool):
        self._model.eval()
        ssims = list()
        psnrs = list()
        if isinstance(self._model, nn.DataParallel):
            model = self._model.module._compressor
        else:
            model = self._model._compressor
        model = model.cuda()
        bs = list()
        zs = list()
        for raw in dataLoader:
            raw = raw.to(self._device, non_blocking=True)

            # restored, _, _, _, _ = self._model(raw, 0.5, True, 0.0)
            latents = model._encoder(raw)
            b, z = model._quantizer.encode(latents, transform)
            bs.append(b[0].detach().cpu())
            zs.append(z[0].detach().cpu())

            quantized = model._quantizer.decode(b, transform)
            restored = model._decoder(quantized)
            raw = self._deTrans(raw)
            restored = self._deTrans(restored)
            ssims.append(evalSSIM(restored.detach(), raw.detach(), True))
            psnrs.append(psnr(restored.detach(), raw.detach()))
        ssims = torch.cat(ssims, 0)
        psnrs = torch.cat(psnrs, 0)
        np.save("b.npy", torch.cat(bs, 0).cpu().numpy())
        # np.save("c.npy", self._model.module.codebook.weight.detach().cpu().numpy())
        np.save("z.npy", torch.cat(zs, 0).cpu().numpy())
        # exit()
        self._logger.info("MS-SSIM: %2.2fdB", ssims.mean())
        self._logger.info("   PSNR: %2.2fdB", psnrs.mean())
        self._saver.add_images("eval/res", restored, global_step=step)
        del bs, zs
        return float(psnrs.mean())
