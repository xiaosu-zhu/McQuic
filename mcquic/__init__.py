from mcquic.consts import Consts
from mcquic.config import Config

from absl import flags


################################ General flags ################################
flags.DEFINE_boolean("debug", False, "Whether to print debug information.", short_name="D")


################################ For `mcquic` ################################
flags.DEFINE_string("input", "", "Input file, should be any images can be read by `torchvision.io.image.decode_image`, or `.bin` file created by mcquic.", short_name="i")
flags.DEFINE_string("output", "", "(Optional) Output file, should be `.png` to restore image, or `.bin` to compress image, or `None` to just print compressor outputs.", short_name="o")
flags.DEFINE_integer("quality", 1, "Target compressor, 1~12, bigger means higher BPP. Now we only release model 1.", short_name="q")


################################ For `mcquic train` ################################
flags.DEFINE_string("config", "", "The `config.yaml` path.", short_name="c")
flags.DEFINE_string("resume", None, "If resume is not None, load checkpoint from path `resume` and resume training", short_name="r")
