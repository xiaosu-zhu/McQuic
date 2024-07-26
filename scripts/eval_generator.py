import torch
import os
import json
import argparse
import cv2
import time
import numpy as np
from PIL import Image

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode, decode_image
from torchvision.transforms.functional import to_pil_image

from mcquic.modules.compressor import BaseCompressor, Compressor, Neon
from mcquic.utils.vision import RandomGamma, RandomPlanckianJitter, RandomAutocontrast, RandomHorizontalFlip, RandomVerticalFlip, PatchWiseErasing
from mcquic.data.transforms import AlignedCrop
from mcquic.validate.handlers import MsSSIM, PSNR
from mcquic.utils.vision import DeTransform
from mcquic.modules.generator_3_var_mcq import GeneratorVARMCQ


def load_model(compressor_path, model_path):
    print("load model...")
    model = GeneratorVARMCQ(
        channel=256,
        k=4096,
        size=[16, 8, 4, 2, 1],
        denseNorm=False,
        loadFrom=compressor_path,
        )
    model.eval().cuda()

    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(args.root, args.dataset)
    # 1. load model
    ms_ssim = MsSSIM().to(0)
    psnr = PSNR().to(0)
    generator = load_model(args.tokenizer_path, args.ckpt)
    
    # 2. load data
    with open("/ssdfs/datahome/tj24011/workspace/McQuic/results/test.txt", "r") as f:
        data = f.readlines()
    
    eval_transform = T.Compose([
        T.ConvertImageDtype(torch.float32),
        AlignedCrop(256),
        T.Normalize(0.5, 0.5),
    ])
    detransform = DeTransform().to(0)
    # dataset = CustomImageDataset(args.inp_path, transform=eval_transform)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    # 3. inference
    print("inference data")
    psnr_res = []
    msssim_res = []
    img_restored = []
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        for idx, item in enumerate(tqdm(data)):
            t0 = time.time()
            samples = generator(None, item)
            t1 = time.time()
            dt = t1 - t0
            
            img = detransform(samples)
            img = img = to_pil_image(img.squeeze(0))
            img.save(f"./0.png")
            
            # for i, sample in enumerate(samples):
            #     img = detransform(sample)
            #     img = to_pil_image(img.squeeze(0))
            #     img.save(f"./{idx}_{i}.png")
            print(f"generated, cost: {dt * 1000}s")

    # 4. calculate metrics
    # mean_psnr = sum(psnr_res) / len(psnr_res)
    # mean_msssim = sum(msssim_res) / len(msssim_res)

    # print(f"PSNR: {mean_psnr}, MS-SSIM: {mean_msssim}")
    
    # 5. save results
    res_path = f"./results/eval/generator/{args.dataset}"
    os.makedirs(res_path, exist_ok=True)
    # for idx, item in enumerate(img_restored):
    #     # cv2_image = np.transpose(item, (1, 2, 0))
    #     # cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    #     # cv2.imwrite(os.path.join(res_path, f"{idx}.png"), cv2_image)
    #     item.save(os.path.join(res_path, f"{idx}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default="results/tokenizers/saved_mcq/val_20000.ckpt")
    parser.add_argument("--tokenizer_path", type=str, default="results/tokenizers/saved_mcq/val_20000.ckpt")
    parser.add_argument("--precision", default="fp32", choices=["bf16", "fp32"])
    parser.add_argument("--hf_token", type=str, default=None, help="huggingface read token for accessing gated repo.")
    parser.add_argument("--dataset", type=str, default="kodak", choices=["kodak", "clic2024"], help="huggingface read token for accessing gated repo.")
    parser.add_argument("--root", type=str, default="/ssdfs/datahome/tj24011/datasets/raw", help="infer data")
    
    args = parser.parse_known_args()[0]
    
    main(args)
