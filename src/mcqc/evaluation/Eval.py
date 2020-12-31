import multiprocessing as mp
from logging import Logger
from typing import Tuple
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from mcqc.datasets import SiftLike, Enumerate
from mcqc.envs import Env
from mcqc.utils import Saver
from mcqc.utils.runtime import Timer
from mcqc.evaluation.helpers import quantizationError, reconstruct
from mcqc import Consts

class Eval:
    def __init__(self, evalOther: bool, savePath: str, dataset: SiftLike, env: Env, model: nn.Module, logger: Logger = None):
        if not evalOther:
            self._model = model.cuda()
            self._env = env
            Saver.Load(savePath, model=self._model)
            self._savePath = savePath
            self._model.eval()
            self._dataset = dataset
        else:
            pass
        torch.autograd.set_grad_enabled(False)

        self._obsMean = None
        self._obsStd = None

        self._logger = logger or Consts.Logger

    def Encode(self, dataset, icm = False, C = None):
        dataloader = DataLoader(dataset, batch_size=100000, shuffle=False, num_workers=mp.cpu_count())
        B = list()
        ticker = Timer()
        for x in tqdm(dataloader, ncols=40, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            x = x.cuda()
            if self._env.DoNormalizationOnObs:
                newX = (x - self._obsMean) / self._obsStd
            b = self._model.Encode(newX, icm=icm, C=C, shift=self._obsMean, scale=self._obsStd)
            B.append(b.detach().cpu())
        interval, _ = ticker.Tick()
        self._logger.info("Encode %d samples for %.2f seconds, %.2e s/sample, %.2e samples/s", len(dataset), interval, interval / len(dataset), len(dataset) / interval)
        B = torch.cat(B, 0)
        return B

    def EncodeFast(self, dataset, icm=True, C = None):
        dataloader = DataLoader(dataset, batch_size=100000, shuffle=False, num_workers=mp.cpu_count())
        B = list()
        icmC = C.reshape(-1, C.shape[-1])
        cUnary = (icmC ** 2).sum(-1)
        cPair = 2 * (icmC @ icmC.t())
        ticker = Timer()
        for x in tqdm(dataloader, ncols=40, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            x = x.cuda()
            if self._env.DoNormalizationOnObs:
                newX = (x - self._obsMean) / self._obsStd
            b = self._model.EncodeFast(newX, icm=True, realX=x, C=icmC, cUnary=cUnary, cPair=cPair)
            B.append(b.detach().cpu())
        interval, _ = ticker.Tick()
        self._logger.info("Encode %d samples for %.2f seconds, %.2e s/sample, %.2e samples/s", len(dataset), interval, interval / len(dataset), len(dataset) / interval)
        B = torch.cat(B, 0)
        return B

    def GetCodebook(self, X, icm=False):
        if self._env.DoNormalizationOnObs:
            newX = X - self._obsMean.to(X.device)
            newX /= self._obsStd.to(X.device)
        B = self._model.Encode(newX)
        C, qError = self._env.Eval(X, B)
        if not icm:
            return C
        B = self._model.Encode(newX, icm=True, C=C, shift=self._obsMean, scale=self._obsStd)
        newC, newQError = self._env.Eval(X, B)
        if newQError.mean() > qError.mean():
            return C
        return newC

    @staticmethod
    def Retrieval(queries, C, B):
        M, _, _ = C.shape
        # if M == 1:
        #     return Eval._retrievalVQ(queryLoader, C, B)

        topK: Tuple[torch.Tensor, torch.Tensor] = None

        N = len(queries)
        ix = torch.arange(len(queries))[:, None]

        baseLoader = DataLoader(Enumerate(B), batch_size=1000, shuffle=False, num_workers=0)
        for i, b in tqdm(baseLoader, ncols=40, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            quantized = reconstruct(C, b)
            # [N, 100]
            dist = ((queries[:, None, ...] - quantized) ** 2).sum(-1)
            i = i.to(b.device).expand_as(dist)
            if topK is not None:
                # [N, 200]
                preparedI = torch.cat([i, topK[0]], -1)
                # [N, 200]
                preparedD = torch.cat([dist, topK[1]], -1)
            else:
                preparedI = i
                preparedD = dist
            # _, [N, 100]
            _, iy = torch.topk(preparedD, k=100, dim=-1, largest=False, sorted=True)
            topK = (preparedI[ix.expand_as(iy), iy], preparedD[ix.expand_as(iy), iy])
        return topK[0]
        # # [NB, D]
        # quantized = reconstruct(C, B)
        # results = []
        # for q in tqdm(queryLoader, ncols=40, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        #     q = q.cuda()
        #     # [NQ, NB]
        #     dist = ((q[:, None, ...] - quantized) ** 2).sum(-1)
        #     _, indices = torch.topk(dist, k=100, dim=-1, largest=False, sorted=True)
        #     results.append(indices.detach())
        # return torch.cat(results, 0)

    @staticmethod
    def RetrievalSlow(queries, C, B):
        M, _, _ = C.shape
        queryLoader = DataLoader(queries, batch_size=1, num_workers=0, shuffle=False)
        if M == 1:
            return Eval._retrievalVQ(queryLoader, C, B)
        # [NB, D]
        quantized = reconstruct(C, B)
        results = []
        for q in tqdm(queryLoader, ncols=40, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            q = q.cuda()
            # [NQ, NB]
            dist = ((q[:, None, ...] - quantized) ** 2).sum(-1)
            _, indices = torch.topk(dist, k=100, dim=-1, largest=False, sorted=True)
            results.append(indices.detach())
        return torch.cat(results, 0)

    @staticmethod
    def _retrievalVQ(queryLoader, C, B):
        # [NB, 1] -> [NB]
        B = B.squeeze()
        results = []
        slot = B.shape[0] // C.shape[1]
        for q in tqdm(queryLoader, ncols=40, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            q = q.cuda()
            # [N, K] <- [N, 1, D], [1, K, D]
            dist = ((q[:, None, ...] - C) ** 2).sum(-1)
            # [N, NB]
            allDist = dist[:, B]
            # [N, slot]
            _, indices = torch.topk(allDist, k=slot, dim=-1, largest=False, sorted=True)
            results.append(indices.detach())
            # [NQ, 100]
        return torch.cat(results, 0)

    @staticmethod
    def Recall(results, groundtruths):
        # [N, 100] = [N, 100] == [N, 1]
        isin = results == groundtruths
        recalls = []
        for i in range(1, isin.shape[-1] + 1):
            recalls.append(isin[:, :i].any(-1).float().mean())
        return torch.Tensor(recalls)

    def Test(self, C=None, B=None):
        sift = self._dataset
        if self._env.DoNormalizationOnObs:
            self._obsMean = sift.data.mean(0).cuda()
            self._obsStd = sift.data.std(0).cuda()
        if C is None:
            sift.Train()
            C = self.GetCodebook(sift.data, icm=True).cuda()
        if B is None:
            sift.Encode(device="cpu")
            B = self.Encode(sift, icm=True, C=C)
            # B = self.EncodeFast(sift, icm=True, C=C)
            self._logger.info("Quantization error in base: %.8e", quantizationError(sift.data.cuda(), C.cuda(), B.cuda()).mean())
        saveDir = os.path.dirname(self._savePath)
        np.save(os.path.join(saveDir, "C.npy"), C.cpu().numpy())
        np.save(os.path.join(saveDir, "B.npy"), B.cpu().numpy().astype(np.uint8))
        self._logger.info("Save C.npy, B.npy at %s", saveDir)
        sift.Query(device="cuda")
        # queryLoader = DataLoader(sift, batch_size=1, shuffle=False, num_workers=mp.cpu_count())
        results = self.Retrieval(sift.data, C.cuda(), B.cuda())
        sift.Gt(device="cuda")
        recalls = self.Recall(results, sift.data[:, :1]) * 100
        self._logger.info("R @ 1: %.2f%%", recalls[0])
        self._logger.info("R @ 10: %.2f%%", recalls[9])
        self._logger.info("R @ 100: %.2f%%", recalls[99])
