from mcqc import ans32
from compressai._CXX import pmf_to_quantized_cdf
import torch

def test():
    a = torch.randint(0, 65536, [2048]).flatten().float()

    prob = a / a.sum()

    cdf = pmf_to_quantized_cdf(prob.tolist(), 16)

    code = torch.randint(0, 2048, [300]).flatten().int().tolist()


    ans32.RansEncoder().encode_with_indexes(code, [0] * 300, [cdf], [2048+2], [0] * 300)


if __name__ == "__main__":
    test()
