
from collections import OrderedDict

from torch import nn

from mcqc import Registry


def build(moduleSpec: OrderedDict):
    layers = list()
    for key, (args, kwArgs) in moduleSpec.items():
        layers.append(
            Registry.get(key)(*args, **kwArgs)
        )
    return nn.Sequential(*layers)
