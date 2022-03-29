from typing import Union, List, Tuple
import torch
from torch import nn

from mcquic import Consts
from mcquic.rans import RansEncoder, RansDecoder
from mcquic.utils.specification import FileHeader, ImageSize, CodeSize
import mcquic

__all__ = [
    "NonNegativeParametrizer",
    "LogExpMinusOne",
    "logExpMinusOne"
]


class _lowerBound(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, input_, bound):
        ctx.save_for_backward(input_, bound)
        return torch.max(input_, bound)

    @staticmethod
    def backward(ctx, grad_output):
        input_, bound = ctx.saved_tensors
        pass_through_if = (input_ >= bound) | (grad_output < 0)
        return pass_through_if.type(grad_output.dtype) * grad_output, None

class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.
    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    def __init__(self, bound: float):
        """Lower bound operator.

        Args:
            bound (float): The lower bound.
        """
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return _lowerBound.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.
    Used for stability during training.
    """
    def __init__(self, minimum: float = 0.0, eps: float = Consts.Eps):
        """Non negative reparametrization.

        Args:
            minimum (float, optional): The lower bound. Defaults to 0.
            reparam_offset (float, optional): Eps for stable training. Defaults to 2**-18.
        """
        super().__init__()

        minimum = float(minimum)
        eps = float(eps)

        self.register_buffer("eps", torch.Tensor([eps ** 2]))
        bound = (minimum + eps ** 2) ** 0.5
        self.lowerBound = LowerBound(bound)

    def init(self, x):
        return torch.sqrt(torch.max(x + self.eps, self.eps))

    def forward(self, x):
        out = self.lowerBound(x)
        out = out ** 2 - self.eps
        return out


class _logExpMinusOne(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return (x.exp() - 1 + Consts.Eps).log()

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        passThroughIf = x > bound
        remaining = ~passThroughIf
        return passThroughIf * grad_output + remaining * grad_output * x.exp() / (x.exp() - 1 + Consts.Eps), None

class LogExpMinusOne(nn.Module):
    def __init__(self, eps: Union[torch.Tensor, float] = Consts.Eps) -> None:
        super().__init__()
        eps = torch.tensor(eps, dtype=torch.float)
        self.register_buffer("_bound", ((1 + eps) / eps).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _logExpMinusOne.apply(x, self._bound)

def logExpMinusOne(x: torch.Tensor, eps: Union[torch.Tensor, float] = Consts.Eps) -> torch.Tensor:
    eps = torch.tensor(eps, dtype=torch.float, device=x.device)
    return _logExpMinusOne.apply(x, ((1 + eps) / eps).log())


def gumbelArgmaxRandomPerturb(logits: torch.Tensor, perturbRate: float = 0.0, tau: float = 1, dim: int = -1) -> torch.Tensor:
    r"""
    Samples from the Gumbel-Argmax distribution and optionally perturb (modify sample to another result).

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      perturbRate: how many samples needs perturbation
      tau: non-negative scalar temperature
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]

    perturbCount = round(index.numel() * perturbRate)
    if perturbCount > 0:
        newIdx = torch.randint(logits.shape[dim], size=(perturbCount, ), device=index.device, dtype=index.dtype)

        # inplace opertaion
        continguous = index.view(-1)
        permIdx = torch.randperm(len(continguous))
        continguous[permIdx[:perturbCount]] = newIdx

    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret

def oneHot(x: torch.LongTensor, numClasses: int, dim: int = -1, dtype = torch.float):
    return torch.zeros((*x.shape, numClasses), dtype=dtype).scatter_(dim, x, 1)


def _checkShape(codes: List[torch.Tensor]):
    info = "Please give codes with correct shape, for example, [[1, 2, 24, 24], [1, 2, 12, 12], ...], which is a `level` length list. each code has shape [n, m, h, w]. "
    if len(codes) < 1:
        raise RuntimeError("Length of codes is 0.")
    n = codes[0].shape[0]
    m = codes[0].shape[1]
    for code in codes:
        newN, newM = code.shape[0], code.shape[1]
        if n < 1:
            raise RuntimeError(info + "Now `n` = 0.")
        if m != newM:
            raise RuntimeError(info + "Now `m` is inconsisitent.")
        if n != newN:
            raise RuntimeError(info + "Now `n` is inconsisitent.")
    return n, m

def compress(encoder: RansEncoder, codes: List[torch.Tensor], ks: List[int], qp: str, imageSize: Tuple[int, int], cdfs: List[List[List[int]]]) -> Tuple[List[List[bytes]], List[FileHeader]]:
    """Compress codes to binary.

    Args:
        codes (List[torch.Tensor]): List of tensor, len = level, code.shape = [n, m, h, w]
        cdfs (List[List[List[int]]]): cdfs for entropy coder, len = level, len(cdfs[0]) = m

    Returns:
        List[List[bytes]]: List of binary, len = n, len(binary[0]) = level
        List[CodeSize]]: List of code size, len = n
    """
    n, m = _checkShape(codes)
    compressed = list(list() for _ in range(n))
    heights = list()
    widths = list()
    # [n, m, h, w]
    for code, ki, cdf in zip(codes, ks, cdfs):
        _, _, h, w = code.shape
        heights.append(h)
        widths.append(w)
        for i, codePerImage in enumerate(code):
            indices = torch.arange(m)[:, None, None]
            # [m, h, w]
            idx = indices.expand_as(codePerImage).flatten().int().tolist()
            cdfSizes = [ki + 2] * m
            # [m, h, w]
            offsets = torch.zeros_like(codePerImage).flatten().int().tolist()
            binary: bytes = encoder.encode_with_indexes(codePerImage.flatten().int().tolist(), idx, cdf, cdfSizes, offsets)
            compressed[i].append(binary)
    header = [FileHeader(mcquic.__version__, qp, CodeSize(m, heights, widths, ks), ImageSize(height=imageSize[0], width=imageSize[1], channel=3)) for _ in range(n)]
    return compressed, header

def decompress(decoder: RansDecoder, binaries: List[List[bytes]], headers: List[FileHeader], cdfs: List[List[List[int]]], device) -> List[torch.Tensor]:
    """Restore codes from binary

    Args:
        binaries (List[List[bytes]]): len = n, len(binary[0]) = level
        codeSizes (List[CodeSize]): len = n
        cdfs (List[List[List[int]]]): len = level, len(cdfs[0]) = m

    Returns:
        List[List[torch.Tensor]]: len = level, each code.shape = [n, m, h, w]
    """
    codeSize = headers[0].CodeSize
    lv = len(binaries[0])
    m = codeSize.m
    codes = list(list() for _ in range(lv))
    indices = torch.arange(m)[:, None, None]
    for binary in binaries:
        # print((codeSize.k, codeSize.heights, codeSize.widths))
        # input()
        for lv, (binaryAtLv, cdf, ki, h, w) in enumerate(zip(binary, cdfs, codeSize.k, codeSize.heights, codeSize.widths)):
            idx = indices.expand(codeSize.m, h, w).flatten().int().tolist()
            cdfSizes = [ki + 2] * codeSize.m
            offsets = torch.zeros(codeSize.m, h, w, dtype=torch.int).flatten().int().tolist()
            restored: List[int] = decoder.decode_with_indexes(binaryAtLv, idx, cdf, cdfSizes, offsets)
            # [m, h, w]
            code = torch.tensor(restored).reshape(codeSize.m, h, w)
            codes[lv].append(code)
    return [torch.stack(c, 0).to(device) for c in codes]
