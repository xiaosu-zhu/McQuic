
from collections import OrderedDict

from torch import nn

from mcqc.layers import BlockFactory


def build(moduleSpec: OrderedDict):
    layers = list()
    for key, (args, kwArgs) in moduleSpec.items():
        layers.append(
            BlockFactory.get(key)(*args, **kwArgs)
        )
    return nn.Sequential(*layers)
