# Copyright NVIDIA/apex
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py

from typing import Union, Optional, Callable

import torch
from torch.utils.data import DataLoader


__all__ = [
    "Prefetcher"
]


class Prefetcher:
    """From `NVIDIA/apex`.

        A DataLoader wrapper that prefetches data for speed-up.
    """
    def __init__(self, loader: DataLoader, rank: int, transform: Optional[Callable] = None):
        """From `NVIDIA/apex`.

            A DataLoader wrapper that prefetches data for speed-up.

        Usage:
        ```python
            trainLoader = DataLoader(dataset, ...)
            # A drop-in replacement for trainLoader
            prefetcher = Prefetcher(trainLoader, rank=dist.get_rank())
            for batch in prefetcher:
                # do something with batch, don't need call .cuda() or .to(device)
                ...
        ```

        Args:
            loader (DataLoader): The DataLoader to be wrapped.
            rank (int): Rank of current DistributedDataParallel.
            transform (Optional[Callable], optional): A transform that is able to accept batched inputs, e.g., `BatchedHorizontalFlip`. Defaults to None.
        """
        self._rank = rank
        self._loader = loader
        self._iter = iter(loader)
        self._stream = torch.cuda.Stream(self._rank) # type: ignore
        self._nextSample: Union[torch.Tensor, None] = None
        self._transform = transform
        self._exhausted = False

    def __iter__(self):
        self._exhausted = False
        self._iter = iter(self._loader)
        return self

    def __next__(self) -> torch.Tensor:
        torch.cuda.current_stream(self._rank).wait_stream(self._stream)
        sample = self._nextSample
        if sample is not None:
            sample.record_stream(torch.cuda.current_stream()) # type: ignore
        else:
            if self._exhausted:
                raise StopIteration
            else:
                self._preLoad()
                torch.cuda.current_stream(self._rank).wait_stream(self._stream)
                sample = self._nextSample
                sample.record_stream(torch.cuda.current_stream()) # type: ignore
        self._preLoad()
        return sample # type: ignore

    def _preLoad(self):
        try:
            sample = next(self._iter)
            with torch.cuda.stream(self._stream):
                # sample = sample.to(self._rank, non_blocking=True)
                if self._transform is not None:
                    sample = self._transform(sample)
                self._nextSample = sample.to(self._rank, non_blocking=True)
        except StopIteration:
            self._nextSample = None
            self._exhausted = True

    def __str__(self) -> str:
        return f"With transform\r\n`{self._transform}`"
