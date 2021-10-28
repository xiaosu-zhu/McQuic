import io
import os
import sys
import urllib
from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc  # pylint:disable=unused-import


# Default URL to fetch metagraphs from.
URL_PREFIX = "https://storage.googleapis.com/tensorflow_compression/metagraphs"
# Default location to store cached metagraphs.
METAGRAPH_CACHE = "/tmp/tfc_metagraphs"


def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  return tf.expand_dims(image, 0)


def write_png(filename, image):
  """Writes a PNG image file."""
  image = tf.squeeze(image, 0)
  if image.dtype.is_floating:
    image = tf.round(image)
  if image.dtype != tf.uint8:
    image = tf.saturate_cast(image, tf.uint8)
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)

def compress_image(model, input_image):
  """Compresses an image tensor into a bitstring."""
  sender = instantiate_model_signature(model, "sender")
  tensors = sender(input_image)
  packed = tfc.PackedTensors()
  packed.model = model
  packed.pack(tensors)
  return packed.string


def compress(model, input_file, output_file, target_bpp=None, bpp_strict=False):
  """Compresses a PNG file to a TFCI file."""
  if not output_file:
    output_file = input_file + ".tfci"

  # Load image.
  input_image = read_png(input_file)
  num_pixels = input_image.shape[-2] * input_image.shape[-3]

  if not target_bpp:
    # Just compress with a specific model.
    bitstring = compress_image(model, input_image)
  else:
    # Get model list.
    models = load_cached(model + ".models")
    models = models.decode("ascii").split()

    # Do a binary search over all RD points.
    lower = -1
    upper = len(models)
    bpp = None
    best_bitstring = None
    best_bpp = None
    while bpp != target_bpp and upper - lower > 1:
      i = (upper + lower) // 2
      bitstring = compress_image(models[i], input_image)
      bpp = 8 * len(bitstring) / num_pixels
      is_admissible = bpp <= target_bpp or not bpp_strict
      is_better = (best_bpp is None or
                   abs(bpp - target_bpp) < abs(best_bpp - target_bpp))
      if is_admissible and is_better:
        best_bitstring = bitstring
        best_bpp = bpp
      if bpp < target_bpp:
        lower = i
      if bpp > target_bpp:
        upper = i
    if best_bpp is None:
      assert bpp_strict
      raise RuntimeError(
          "Could not compress image to less than {} bpp.".format(target_bpp))
    bitstring = best_bitstring

  # Write bitstring to disk.
  with tf.io.gfile.GFile(output_file, "wb") as f:
    f.write(bitstring)


def decompress(input_file, output_file):
  """Decompresses a TFCI file and writes a PNG file."""
  if not output_file:
    output_file = input_file + ".png"
  with tf.io.gfile.GFile(input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  receiver = instantiate_model_signature(packed.model, "receiver")
  tensors = packed.unpack([t.dtype for t in receiver.inputs])
  output_image, = receiver(*tensors)
  write_png(output_file, output_image)



def list_models():
  """Lists available models in web storage with a description."""
  url = URL_PREFIX + "/models.txt"
  request = urllib.request.urlopen(url)
  try:
    print(request.read().decode("utf-8"))
  finally:
    request.close()


def list_tensors(model):
  """Lists all internal tensors of a given model."""
  def get_names_dtypes_shapes(function):
    for op in function.graph.get_operations():
      for tensor in op.outputs:
        yield tensor.name, tensor.dtype.name, tensor.shape

  sender = instantiate_model_signature(model, "sender")
  tensors = sorted(get_names_dtypes_shapes(sender))
  print("Sender-side tensors:")
  for name, dtype, shape in tensors:
    print(f"{name} (dtype={dtype}, shape={shape})")
  print()

  receiver = instantiate_model_signature(model, "receiver")
  tensors = sorted(get_names_dtypes_shapes(receiver))
  print("Receiver-side tensors:")
  for name, dtype, shape in tensors:
    print(f"{name} (dtype={dtype}, shape={shape})")


def dump_tensor(model, tensors, input_file, output_file):
  """Dumps the given tensors of a model in .npz format."""
  if not output_file:
    output_file = input_file + ".npz"
  # Note: if receiver-side tensors are requested, this is no problem, as the
  # metagraph contains the union of the sender and receiver graphs.
  sender = instantiate_model_signature(model, "sender", outputs=tensors)
  input_image = read_png(input_file)
  # Replace special characters in tensor names with underscores.
  table = str.maketrans(r"^./-:", r"_____")
  tensors = [t.translate(table) for t in tensors]
  values = [t.numpy() for t in sender(input_image)]
  assert len(tensors) == len(values)
  # Write to buffer first, since GFile might not be random accessible.
  with io.BytesIO() as buf:
    np.savez(buf, **dict(zip(tensors, values)))
    with tf.io.gfile.GFile(output_file, mode="wb") as f:
      f.write(buf.getvalue())




def main(args):
  # Command line can override these defaults.
  global URL_PREFIX, METAGRAPH_CACHE
  URL_PREFIX = args.url_prefix
  METAGRAPH_CACHE = args.metagraph_cache

  # Invoke subcommand.
  if args.command == "compress":
    compress(args.model, args.input_file, args.output_file,
             args.target_bpp, args.bpp_strict)
  elif args.command == "decompress":
    decompress(args.input_file, args.output_file)
  elif args.command == "models":
    list_models()
  elif args.command == "tensors":
    list_tensors(args.model)
  elif args.command == "dump":
    if not args.tensor:
      raise ValueError("Must provide at least one tensor to dump.")
    dump_tensor(args.model, args.tensor, args.input_file, args.output_file)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
