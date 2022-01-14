import json
import os
from typing import Dict

import numpy as np
from tqdm import tqdm, trange
from vlutils.config import read
from mcqc import Config

from absl import app
from absl import flags

import torch
from torch import nn

from mcqc.evaluation.refModel import PostProcess, Preprocess, RefDecoder5x5, RefEncoder, RefDecoder, RefEncoder5x5

FLAGS = flags.FLAGS


def convert(preProcess: nn.Module, encoder: nn.Module, decoder: nn.Module, postProcess: nn.Module):
    preProcess = preProcess.eval().cuda()
    encoder = encoder.eval().cuda()
    decoder = decoder.eval().cuda()
    postProcess = postProcess.eval().cuda()

    with torch.jit.optimized_execution(True):
        scriptedEncoder = torch.jit.script(encoder)
        scriptedDecoder = torch.jit.script(decoder)
        return scriptedEncoder, scriptedDecoder


class Speed():
    def __init__(self, preProcess, postProcess, encoder, decoder) -> None:
        self._preProcess = preProcess
        self._postProcess = postProcess
        self._encoder = encoder
        self._decoder = decoder
        # same as kodak
        self._testInput = torch.rand(6, 3, 768, 512).cuda()
        self._warmupStep = 10
        self._evalStep = 100

    def test(self):
        x, cAndPadding = self._preProcess(self._testInput)
        # warmup
        for _ in trange(self._warmupStep, leave=False):
            b, cAndPadding = self._encoder(x, cAndPadding)
            self._decoder(b, cAndPadding)
        torch.cuda.synchronize()

        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent = torch.cuda.Event(enable_timing=True)


        startEvent.record()
        # test encoder
        for _ in trange(self._evalStep, leave=False):
            self._encoder(x, cAndPadding)
        endEvent.record()
        torch.cuda.synchronize()
        encoderMs = startEvent.elapsed_time(endEvent) / (self._evalStep * len(self._testInput))

        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent = torch.cuda.Event(enable_timing=True)

        startEvent.record()
        # test encoder
        for _ in trange(self._evalStep, leave=False):
            self._decoder(b, cAndPadding)
        endEvent.record()
        torch.cuda.synchronize()
        decoderMs = startEvent.elapsed_time(endEvent) / (self._evalStep * len(self._testInput))

        return {"encoderForwardTime": encoderMs, "decoderForwardTime": decoderMs}



def _main(encoder, decoder):
    torch.autograd.set_grad_enabled(False)
    encoder, decoder = convert(Preprocess(128), encoder, decoder, PostProcess())
    result = Speed(Preprocess(128), PostProcess(), encoder, decoder).test()
    del encoder, decoder
    return result["encoderForwardTime"], result["decoderForwardTime"]


def testDifferentCodebooks():
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    ms = [1, 2, 4, 6, 8, 12, 16, 24]
    ks = list(range(6, 14))
    results = np.zeros([len(ms), len(ks), 2])
    for i, m in enumerate(ms):
        for j, k in enumerate(ks):
            result = _main(RefEncoder(m, [int(2 ** k)], 192, 1, False), RefDecoder(m, [int(2 ** k)], 192, False))
            results[i, j, 0] = result[0]
            results[i, j, 1] = result[1]
            torch.cuda.empty_cache()
    np.save(f"testAllSizes_m_{ms[0]}_{ms[-1]}_k_{int(2 ** ks[0])}_{int(2 ** ks[-1])}.npy", results)



class RoundingModel(nn.Module):
    def forward(self, x):
        return x.int()


class LookupModel(nn.Module):
    def __init__(self, m: int, k: int, d: int):
        super().__init__()
        self._m = m
        self._codebook = nn.Parameter(torch.empty(m, k, d))
        self.register_buffer("_ix", torch.arange(m))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        # use codes to index codebook (m, k, d) ==> [n, h, w, m, k] -> [n, c, h, w]
        # ix = torch.arange(self._m, device=codes.device).expand_as(codes)
        ix = self._ix.expand_as(codes)
        # [n, h, w, m, d]
        indexed = self._codebook[ix, codes]
        return indexed
        # return self.decode(codes)


def testDifferentCodebooks():
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    ms = [1, 2, 4, 6, 8, 12, 16, 24]
    ks = list(range(6, 14))
    results = np.zeros([len(ms), len(ks), 2])
    for i, m in enumerate(ms):
        for j, k in enumerate(ks):
            result = _main(RefEncoder(m, [int(2 ** k)], 192, 1, False), RefDecoder(m, [int(2 ** k)], 192, False))
            results[i, j, 0] = result[0]
            results[i, j, 1] = result[1]
            torch.cuda.empty_cache()
    np.save(f"testAllSizes_m_{ms[0]}_{ms[-1]}_k_{int(2 ** ks[0])}_{int(2 ** ks[-1])}.npy", results)


def testRoundingVSLookup():
    warmupStep = 100
    testStep = 1000
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)

    lookUpResults = list()

    H = 512
    L = 64
    N = 192

    for m in [1, 2, 4, 6, 8, 12, 16, 24]:

        b = torch.zeros(L, H, H, m, dtype=int, device="cuda")
        # K does not affect latency
        lookUp = LookupModel(m, 16, N // m).eval().cuda()

        with torch.jit.optimized_execution(True):
            lookUp = torch.jit.script(lookUp)

            # warmup
            for _ in trange(warmupStep, leave=False):
                lookUp(b)
            torch.cuda.synchronize()

            startEvent = torch.cuda.Event(enable_timing=True)
            endEvent = torch.cuda.Event(enable_timing=True)

            startEvent.record()
            # test encoder
            for _ in trange(testStep, leave=False):
                lookUp(b)
            endEvent.record()
            torch.cuda.synchronize()
            lookupMS = startEvent.elapsed_time(endEvent) / (testStep * L)
        lookUpResults.append(lookupMS)
        del lookUp, b
        torch.cuda.empty_cache()

    x = torch.zeros(L, N, H, H, device="cuda")

    rounding = RoundingModel().eval().cuda()

    with torch.jit.optimized_execution(True):
        rounding = torch.jit.script(rounding)


        # warmup
        for _ in trange(warmupStep, leave=False):
            rounding(x)
        torch.cuda.synchronize()

        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent = torch.cuda.Event(enable_timing=True)

        startEvent.record()
        # test encoder
        for _ in trange(testStep, leave=False):
            rounding(x)
        endEvent.record()
        torch.cuda.synchronize()
        roundingMS = startEvent.elapsed_time(endEvent) / (testStep * L)

    del rounding, x
    torch.cuda.empty_cache()

    return roundingMS, lookUpResults

if __name__ == "__main__":
    # testDifferentCodebooks()
    rounding, lookup = testRoundingVSLookup()
    with open("rounding_vs_lookup.json", "w") as fp:
        json.dump({
            "rounding": rounding,
            "lookup": lookup
        }, fp)
