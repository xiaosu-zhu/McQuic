import os
from typing import Dict

import torch
from torch import nn

from mcqc.evaluation.refModel import RefEncoder, RefDecoder


def migrate(model: nn.Module, prefix: str, stateDict: Dict[str, torch.Tensor]):
    stateDict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in stateDict.items()}
    keys = model.keys
    stateDict = {k: v for k, v in stateDict.items() if k.startswith(keys)}
    model.load_state_dict(stateDict)
    return model


def convert(encoder: nn.Module, decoder: nn.Module, path: str):
    testInput = torch.rand([1, 3, 1357, 2468])
    try:
        testOutput = decoder(*encoder(testInput))
        if testInput.shape != testOutput.shape:
            raise RuntimeError(f"Input-output shape mismatch, expected {testInput.shape}, got {testOutput.shape}, please check model.")
        if torch.any(testOutput < 0) or torch.any(testOutput > 1):
            raise RuntimeError("Find result out of range 0~1, please check model.")
    except Exception as e:
        raise RuntimeError(f"Model error with input {testInput.shape}.") from e
    testInput = torch.rand([1, 3, 2, 2])
    try:
        testOutput = decoder(*encoder(testInput))
        if testInput.shape != testOutput.shape:
            raise RuntimeError(f"Input-output shape mismatch, expected {testInput.shape}, got {testOutput.shape}, please check model.")
        if torch.any(testOutput < 0) or torch.any(testOutput > 1):
            raise RuntimeError("Find result out of range 0~1, please check model.")
    except Exception as e:
        raise RuntimeError(f"Model error with input {testInput.shape}.") from e
    scriptedEncoder = torch.jit.script(encoder)
    scriptedEncoder.save(os.path.join(path, "encoder.pt"))
    scriptedDecoder = torch.jit.script(decoder)
    scriptedDecoder.save(os.path.join(path, "decoder.pt"))


def main(encoder, decoder, prefix, ckptPath, targetDir):
    torch.autograd.set_grad_enabled(False)
    stateDict = torch.load(ckptPath, map_location={"cuda:0": "cpu"})["model"]

    encoder = migrate(encoder, prefix, stateDict)
    decoder = migrate(decoder, prefix, stateDict)

    convert(encoder, decoder, targetDir)


if __name__ == "__main__":
    main(RefEncoder(32, 256, 256), RefDecoder(32, 256, 256), "module._compressor.", "ckpt/global.ckpt", "ckpt")
