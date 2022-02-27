from absl import app, flags


FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input file, should be any images can be read by `torchvision.io.image.decode_image`, or `.bin` file created by mcquic.", short_name="i")
flags.DEFINE_string("output", "", "(Optional) Output file, should be `.png` to restore image, or `.bin` to compress image, or `None` to just print compressor outputs.")
flags.DEFINE_integer("quality", 1, "Target compressor, 1~12, bigger means higher BPP. Now we only release model 1.", short_name="q")
flags.DEFINE_boolean("debug", False, "Whether to print debug information.", short_name="D")


def main(_):
    pass


if __name__ == "__main__":
    app.run(main)
