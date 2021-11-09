import os
import subprocess
from pathlib import Path


from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("source", "kodak/kodim01.png", "The source png file.")
flags.DEFINE_string("target", "result", "The target dir.")

flags.DEFINE_string("bpg", "4,47", "The -m param and -q param of bpgenc.")
flags.DEFINE_string("vvc", "45", "The -q param of VVC.")

def main(_):
    filename = Path(FLAGS.source).stem
    print("================ BPG ================")
    m, q = FLAGS.bpg.strip().split(",")
    process = subprocess.Popen(f"bpgenc -m {m} -q {q} {FLAGS.source} -o {filename}.bpg", shell=True, stdout=subprocess.PIPE)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Run `bpgenc` with error {process.returncode}.")
