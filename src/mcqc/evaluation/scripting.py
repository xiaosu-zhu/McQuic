import os
from typing import Dict

import torch
from torch import nn


def migrate(model: nn.Module, modelName: str, prefix: str, stateDict: Dict[str, torch.Tensor]):
    stateDict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in stateDict.items()}
    stateDict = {k: v for k, v in stateDict.items() if k.startswith(modelName)}
    model.load_state_dict(stateDict)
    return model


def convert(model: nn.Module, path: str):
    testInput = torch.rand([1, 3, 1357, 2468])
    testOutput = model(testInput)
    if testInput.shape != testOutput.shape:
        raise RuntimeError(f"Input-output shape mismatch, expected {testInput.shape}, got {testOutput.shape}, please check model.")
    if any(testOutput < 0) or any(testOutput > 1):
        raise RuntimeError("Find output out of range 0~1, please check model.")
    scriptedModel = torch.jit.script(model)
    scriptedModel.save(path)


def main(model, ckptPath, targetDir):
    stateDict = torch.load(ckptPath)["model"]
    prefix = model.getPrefix()

    encoder, encoderName = model.getEncoder()
    encoder = migrate(encoder, encoderName, prefix, stateDict)
    decoder, decoderName = model.getDecoder()
    decoder = migrate(decoder, decoderName, prefix, stateDict)

    convert(encoder, os.path.join(targetDir, "encoder.pt"))
    convert(decoder, os.path.join(targetDir, "decoder.pt"))


if __name__ == "__main__":
    main(None, None, None)
