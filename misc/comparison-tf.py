import io
import time
import os
import sys
import urllib
from urllib import request

from tqdm import trange
from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc  # pylint:disable=unused-import


# Default URL to fetch metagraphs from.
URL_PREFIX = "https://storage.googleapis.com/tensorflow_compression/metagraphs"
# Default location to store cached metagraphs.
METAGRAPH_CACHE = "/tmp/tfc_metagraphs"


balle_factorized = "bmshj2018-factorized-msssim-6"
balle_hyperprior = "bmshj2018-hyperprior-msssim-6"
minnen2018 = "mbt2018-mean-msssim-6"

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


def instantiate_model_signature(model, signature, inputs=None, outputs=None):
  """Imports a trained model and returns one of its signatures as a function."""
  string = load_cached(model + ".metagraph")
  metagraph = tf.compat.v1.MetaGraphDef()
  metagraph.ParseFromString(string)
  wrapped_import = tf.compat.v1.wrap_function(
      lambda: tf.compat.v1.train.import_meta_graph(metagraph), [])
  graph = wrapped_import.graph
  if inputs is None:
    inputs = metagraph.signature_def[signature].inputs
    inputs = [graph.as_graph_element(inputs[k].name) for k in sorted(inputs)]
  else:
    inputs = [graph.as_graph_element(t) for t in inputs]
  if outputs is None:
    outputs = metagraph.signature_def[signature].outputs
    outputs = [graph.as_graph_element(outputs[k].name) for k in sorted(outputs)]
  else:
    outputs = [graph.as_graph_element(t) for t in outputs]
  return wrapped_import.prune(inputs, outputs)


def compress_image(model, input_image):
  """Compresses an image tensor into a bitstring."""
  sender = instantiate_model_signature(model, "sender")
  # warm up
  for i in range(5):
    tensors = sender(input_image)
  start = time.time()
  for i in trange(500):
    tensors = sender(input_image)
  end = time.time()
  print("encoder time in ms:", (end - start) / 500 * 1000)
  packed = tfc.PackedTensors()
  packed.model = model
  packed.pack(tensors)
  return packed.string


def load_cached(filename):
  """Downloads and caches files from web storage."""
  pathname = os.path.join(METAGRAPH_CACHE, filename)
  try:
    with tf.io.gfile.GFile(pathname, "rb") as f:
      string = f.read()
  except tf.errors.NotFoundError:
    url = f"{URL_PREFIX}/{filename}"
    request = urllib.request.urlopen(url)
    try:
      string = request.read()
    finally:
      request.close()
    tf.io.gfile.makedirs(os.path.dirname(pathname))
    with tf.io.gfile.GFile(pathname, "wb") as f:
      f.write(string)
  return string



def compress(model, input_file, output_file):
  """Compresses a PNG file to a TFCI file."""
#   output_file = input_file + ".tfci"

  # Load image.
  input_image = read_png(input_file)
  num_pixels = input_image.shape[-2] * input_image.shape[-3]

  # Just compress with a specific model.
  bitstring = compress_image(model, input_image)
  # Write bitstring to disk.
  with tf.io.gfile.GFile(output_file, "wb") as f:
    f.write(bitstring)


def decompress(model, input_file, output_file):
  """Decompresses a TFCI file and writes a PNG file."""
  if not output_file:
    output_file = input_file + ".png"
  with tf.io.gfile.GFile(input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  receiver = instantiate_model_signature(model, "receiver")
  tensors = packed.unpack([t.dtype for t in receiver.inputs])
  # warm up
  for i in range(5):
    output_image, = receiver(*tensors)
  start = time.time()
  for i in trange(500):
    output_image, = receiver(*tensors)
  end = time.time()
  print("decoder time in ms:", (end - start) / 500 * 1000)
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
#   URL_PREFIX = args.url_prefix
#   METAGRAPH_CACHE = args.metagraph_cache

  compress(minnen2018, "data/kodak/kodim01.png", "tmp/kodim01.tfci")
  decompress(minnen2018, "tmp/kodim01.tfci", "tmp/kodim01.png")


if __name__ == "__main__":
  app.run(main)
