from typing import Tuple
from absl import flags

from mcquic.utils import EntrypointRegistry

FLAGS = flags.FLAGS


""" For `mcquic`
flags.DEFINE_string("input", "", "Input file, should be any images can be read by `torchvision.io.image.decode_image`, or `.bin` file created by mcquic.", short_name="i")
flags.DEFINE_string("output", "", "(Optional) Output file, should be `.png` to restore image, or `.bin` to compress image, or `None` to just print compressor outputs.")
flags.DEFINE_integer("quality", 1, "Target compressor, 1~12, bigger means higher BPP. Now we only release model 1.", short_name="q")
"""


lazyLoads = {
    "train": "mcquic.train.cli"
}


def dispatch(args: Tuple[str, ...]) -> int:
    if len(args) == 1:
        return _main(args)

    if args[1] in lazyLoads:
        import importlib
        importlib.import_module(lazyLoads[args[1]])

    entrypoint = EntrypointRegistry.get(args[1])

    if entrypoint is None:
        availableEntrypoints = ", ".join(f"`mcquic {x}`" for x in EntrypointRegistry._map.keys())
        raise ValueError(f"No entrypoint for `mcquic {args[1]}`. Avaliable entrypoints: `mcquic`, {availableEntrypoints}")

    return entrypoint(args)


def _main(_) -> int:
    return 0

def main(args: Tuple[str, ...]) -> int:
    return dispatch(args)
