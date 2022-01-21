import torch
import trtorch


compileSettings = {
    "inputs": [trtorch.Input(
        min_shape=[1, 3, 128, 128],
        opt_shape=[1, 3, 768, 1024],
        max_shape=[1, 3, 4096, 4096],
        dtype=torch.float
    ), trtorch.Input(shape=[4], dtype=torch.int)]
}

scripttedEncoder = torch.jit.load("ckpt/encoder.pt")

tensorRTEncoder = trtorch.compile(scripttedEncoder, compileSettings)

x = torch.rand(1, 3, 768, 768)

print(tensorRTEncoder(x))
