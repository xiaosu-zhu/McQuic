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
from mcqc import Consts

def _ssimExp(source, target, datarange):
    return (2.7182818284590452353602874713527 - ms_ssim(source, target, data_range=datarange).exp()) / (1.7182818284590452353602874713527)


class Plain(Algorithm):
    def __init__(self, model: nn.Module, device: str, optimizer: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer], scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler], saver: Saver, continueTrain: bool, logger: Logger, epoch: int):
        super().__init__()
        self._model = model
        if device == "cuda" and torch.cuda.device_count() > 1:
            self._model = nn.DataParallel(self._model.to(device))
        else:
            self._model = self._model.to(device)
        self._device = device
        self._optimizer = optimizer(self._model.parameters())
        self._scheduler = scheduler(self._optimizer)
        self._epoch = epoch
        self._saver = saver
        self._logger = logger

        self._continue = continueTrain

    @staticmethod
    def _deTrans(imaage):
        return ((imaage * 0.5 + 0.5) * 255).clamp(0.0, 255.0).byte()

    def run(self, trainLoader: torch.utils.data.DataLoader, testLoader: torch.utils.data.DataLoader):
        initTemp = 1.0
        minTemp = 0.5
        step = 0
        mixin = 1.0

        if self._continue:
            loaded = self._saver.load(self._saver.SavePath, self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, temp=initTemp, mixin=mixin)
            initTemp = loaded["temp"]
            step = loaded["step"]
            mixin = loaded["mixin"]

        for i in range(self._epoch):
            self._model.train()
            for images in trainLoader:
                images = images.to(self._device, non_blocking=True)
                hsvImages = (rgb2hsv((images + 1.) / 2.) - 0.5) / 0.5
                restored, codes, latents, logits, quantizeds, targets = self._model(torch.cat([images, hsvImages], axis=1), initTemp, True, mixin)
                # print(quantizeds[0][0, :3, 4:10, 4:10])
                loss = self._loss(step, images, hsvImages, restored, codes, latents, logits, quantizeds, targets)
                self._model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=10.0)
                self._optimizer.step()
                self._saver.add_scalar("loss", loss, global_step=step)
                if (step + 1) % 100 == 0:
                    self._saver.add_images("soft/raw", self._deTrans(images), global_step=step)
                    self._saver.add_images("soft/res", self._deTrans(restored[:, 3:]), global_step=step)
                    self._saver.add_histogram("code", codes[0].argmax(-1), global_step=step)
                if (step + 1) % 1000 == 0:
                    self._eval(testLoader, step)
                    self._saver.save(self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, temp=initTemp, mixin=mixin)
                    self._logger.info("%3dk steps complete, update: LR = %.2e, T = %.2e, M = %.2e", (step + 1) // 1000, self._scheduler.get_last_lr()[0], initTemp, mixin)
                if (step + 1) % 10000 == 0:
                    self._scheduler.step()
                step += 1
                mixin *= 0.995

    @torch.no_grad()
    def _eval(self, dataLoader: torch.utils.data.DataLoader, step: int):
        self._model.eval()
        ssims = list()
        psnrs = list()
        if isinstance(self._model, nn.DataParallel):
            model = self._model.module
        else:
            model = self._model
        model = model.cuda()
        for raw in dataLoader:
            raw = raw.to(self._device, non_blocking=True)
            hsvRaw = (rgb2hsv((raw + 1.) / 2.) - 0.5) / 0.5

            restored, codes, latents, logits, quantizeds, targets = self._model(torch.cat([raw, hsvRaw], axis=1), 0.5, True, 0.0)
            # latents = model._encoder(torch.cat([raw, hsvRaw], axis=1))
            # b = model._quantizer.encode(latents)
            # quantized = model._quantizer.decode(b)
            # restored = model._decoder(quantized)
            raw = self._deTrans(raw)
            restored = self._deTrans(restored[:, 3:])
            ssims.append(evalSSIM(restored.detach(), raw.detach(), True))
            psnrs.append(psnr(restored.detach(), raw.detach()))
        ssims = torch.cat(ssims, 0)
        psnrs = torch.cat(psnrs, 0)
        self._logger.info("MS-SSIM: %2.2fdB", ssims.mean())
        self._logger.info("   PSNR: %2.2fdB", psnrs.mean())
        self._saver.add_images("eval/res", restored, global_step=step)

    def _loss(self, step, images, hsvImages, restored, codes, latents, logits, quantizeds, targets):
        hsvR, rgbR = torch.chunk(restored, 2, 1)
        # combined = torch.cat([images, hsvImages], axis=1)
        l2Loss = torch.nn.functional.mse_loss(rgbR, images) + torch.nn.functional.mse_loss(hsvR, hsvImages)
        l1Loss = torch.nn.functional.l1_loss(rgbR, images) + torch.nn.functional.l1_loss(hsvR, hsvImages)
        ssimLoss = 2 - ms_ssim((rgbR + 1), (images + 1), data_range=2.0) - ms_ssim((hsvR + 1), (hsvImages + 1), data_range=2.0)
        self._saver.add_scalar("loss/l2", l2Loss, global_step=step)
        self._saver.add_scalar("loss/l1", l1Loss, global_step=step)
        self._saver.add_scalar("loss/ssim", ssimLoss, global_step=step)

        transformerL2 = list()
        transformerL1 = list()
        for q, t in zip(quantizeds, targets):
            transformerL2.append(torch.nn.functional.mse_loss(q, t))
            transformerL1.append(torch.nn.functional.l1_loss(q, t))
        transformerL2 = sum(transformerL2)
        transformerL1 = sum(transformerL1)


        self._saver.add_scalar("loss/tl2", transformerL2, global_step=step)
        self._saver.add_scalar("loss/tl1", transformerL2, global_step=step)

        # ssimLoss = _ssimExp((rgbR + 1), (images + 1), 2.0) + _ssimExp((hsvR + 1), (hsvImages + 1), 2.0)

        regs = list()
        for logit in logits:
            # N, H, W, K -> NHW, K
            logit = logit.reshape(-1, logit.shape[-1])
            # distributions = Categorical(logit.permute(0, 2, 3, 1).reshape(-1, logit.shape[1]))
            # entropies.append(distributions.entropy().mean())
            reg = compute_penalties(logit, global_entropy_coeff=0.01, cv_coeff=0.01, eps=Consts.Eps)
            regs.append(reg)

        regs = sum(regs)

        self._saver.add_scalar("loss/reg", regs, global_step=step)

        # klLoss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logitsConsistency, -1), torch.nn.functional.log_softmax(logitsCompressed.detach(), -1), reduction="batchmean", log_target=True)
        # rgbRhsv = (rgb2hsv((rgbR + 1.) / 2.) - 0.5) / 0.5
        # hsvRrgb = (hsv2rgb((hsvR + 1.) / 2.) - 0.5) / 0.5
        # crossL2Loss = torch.nn.functional.mse_loss(hsvRrgb, images) + torch.nn.functional.mse_loss(rgbRhsv, hsvImages)
        # crossSSIM = 2 - ms_ssim((hsvRrgb + 1), (images + 1), data_range=2.0) - ms_ssim((rgbRhsv + 1), (hsvImages + 1), data_range=2.0)
        return l2Loss \
             + l1Loss \
             + ssimLoss \
             + 1. * regs \
           # + transformerL1 + transformerL2 \
           # + 1e-3 * (crossL2Loss + crossSSIM) \
           # + 1e-6 * klLoss \



def compute_penalties(logits, individual_entropy_coeff=0.0, allowed_entropy=0.0, global_entropy_coeff=0.0,
                      cv_coeff=0.0, eps=1e-9):
    """
    Computes typical regularizers for gumbel-softmax quantization
    Regularization is of slight help when performing hard quantization, but it isn't critical
    :param logits: tensor [batch_size, ..., codebook_size]
    :param individual_entropy_coeff: penalizes mean individual entropy
    :param allowed_entropy: does not penalize individual_entropy if it is below this value
    :param cv_coeff: penalizes squared coefficient of variation
    :param global_entropy_coeff: coefficient for entropy of mean probabilities over batch
        this value should typically be negative (e.g. -1), works similar to cv_coeff
    """
    p = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    # [batch_size, ..., codebook_size]

    individual_entropy_values = -torch.sum(p * logp, dim=-1)
    clipped_entropy = torch.nn.functional.relu(allowed_entropy - individual_entropy_values + eps).mean()
    individual_entropy = (individual_entropy_values.mean() - clipped_entropy).detach() + clipped_entropy

    global_p = torch.mean(p, dim=0)  # [..., codebook_size]
    global_logp = torch.logsumexp(logp, dim=0) - np.log(float(logp.shape[0]))  # [..., codebook_size]
    global_entropy = -torch.sum(global_p * global_logp, dim=-1).mean()

    load = torch.mean(p, dim=0)  # [..., codebook_size]
    mean = load.mean()
    variance = torch.mean((load - mean) ** 2)
    cvPenalty = variance / (mean ** 2 + eps)
    return individual_entropy_coeff * individual_entropy + global_entropy_coeff * global_entropy + cv_coeff * cvPenalty
