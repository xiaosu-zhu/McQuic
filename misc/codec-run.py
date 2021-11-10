import os
import subprocess
from pathlib import Path
import math

from absl import app
from absl import flags
from PIL import Image
from tabulate import tabulate
import glymur
import cv2

import torch
import torchvision
from mcqc.evaluation.metrics import psnr, MsSSIM


FLAGS = flags.FLAGS

flags.DEFINE_string("source", "kodak/kodim01.png", "The source png file.")
flags.DEFINE_string("target", "result", "The target dir.")
flags.DEFINE_string("vvc-cfg", "~/VVC/cfg/encoder_intra_vtm.cfg", "The VVC bin directory.")

flags.DEFINE_string("bpg", "4,47", "The -m param and -q param of bpgenc.")
flags.DEFINE_string("jp2", "100,90,80", "The -m param and -q param of bpgenc.")
flags.DEFINE_string("vvc", "43", "The -q param of VVC.")

def run(command: str):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Run `{command.strip().split()[0]}` with error {process.returncode}.")


def pprint(notify, ncol=100):
    leftCol = (ncol - 2 - len(notify)) // 2
    rightCol = ncol - 2 - len(notify) - leftCol
    print("\n" + "=" * leftCol + f" {notify} " + "=" * rightCol + "\n")


def checkStat(binFile, rawPNG, compressedPNG):
    fileSize = os.path.getsize(binFile)


    rawTensor = torchvision.io.read_image(rawPNG).float().unsqueeze(0)

    compressedTensor = torchvision.io.read_image(compressedPNG).float().unsqueeze(0)

    _, _, h, w = compressedTensor.shape

    return {
        "psnr": psnr(rawTensor, compressedTensor).item(),
        "ssim": -10 * math.log10(1 - MsSSIM()(rawTensor, compressedTensor).item()),
        "bpp": (fileSize * 8) / (h * w)
    }

def main(_):

    table = []

    os.makedirs(FLAGS.target, exist_ok=True)
    filename = Path(FLAGS.source).stem


    pprint("JPEG-2000")
    ext = ".jp2"
    targetJP2 = os.path.join(FLAGS.target, filename + ext)
    targetPNG = os.path.join(FLAGS.target, "jp2-" + filename + ".png")
    cratios = [int(x) for x in FLAGS.jp2.strip().split(",")]

    image = cv2.imread(FLAGS.source)
    jp2 = glymur.Jp2k(targetJP2, data=image, cratios=cratios)

    jp2 = glymur.Jp2k(targetJP2)
    data = jp2[:]

    cv2.imwrite(targetPNG, data)

    result = checkStat(targetJP2, FLAGS.source, targetPNG)

    table.append(["JPEG-2000", result["psnr"], result["ssim"], result["bpp"]])


    pprint("BPG")
    ext = ".bpg"
    targetBPG = os.path.join(FLAGS.target, filename + ext)
    targetPNG = os.path.join(FLAGS.target, "bpg-" + filename + ".png")
    m, q = FLAGS.bpg.strip().split(",")
    run(f"bpgenc -m {m} -q {q} {FLAGS.source} -o {targetBPG}")
    run(f"bpgdec {targetBPG} -o {targetPNG}")

    result = checkStat(targetBPG, FLAGS.source, targetPNG)

    table.append(["BPG", result["psnr"], result["ssim"], result["bpp"]])

    pprint("VVC")

    image = Image.open(FLAGS.source)
    w, h = image.size

    cfgLocation = FLAGS.get_flag_value("vvc-cfg", "")
    ext = ".bin"
    targetYUV = os.path.join(FLAGS.target, filename + ".yuv")
    targetVVC = os.path.join(FLAGS.target, filename + ext)
    targetOUT = os.path.join(FLAGS.target, "VVC-" + filename + ".yuv")
    targetPNG = os.path.join(FLAGS.target, "VVC-" + filename + ".png")
    q = FLAGS.vvc.strip()
    pprint("VVC: Convert to YUV")
    run(f"ffmpeg -y -i {FLAGS.source} -pix_fmt yuv444p {targetYUV}")
    pprint("VVC: Compress and restore")
    run(f"EncoderApp -c {cfgLocation} -i {targetYUV} -b {targetVVC} -o {targetOUT} -f 1 -fr 2 -wdt {w} -hgt {h} -q {q} --OutputBitDepth=8 --OutputBitDepthC=8 --InputChromaFormat=444")
    pprint("VVC: Back to PNG")
    run(f"ffmpeg -y -s {w}x{h} -r 2 -pixel_format yuv444p -i {targetOUT} {targetPNG}")

    result = checkStat(targetVVC, FLAGS.source, targetPNG)

    table.append(["VVC", result["psnr"], result["ssim"], result["bpp"]])


    pprint("Summary")


    print(tabulate(table, headers=["Methods", "PSNR", "MS-SSIM", "BPP"]))


if __name__ == "__main__":
    app.run(main)




# python misc/codec-run.py --source data/kodak/kodim07.png --target tmp -bpg 4,43 -vvc 39 -jp2 180
# python misc/codec-run.py --source data/kodak/kodim01.png --target tmp -bpg 5,46 -vvc 42 -jp2 180
# python misc/codec-run.py --source data/kodak/kodim24.png --target tmp -bpg 5,46 -vvc 41 -jp2 180
