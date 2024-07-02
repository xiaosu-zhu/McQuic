import torch
import os
import json
import argparse
import cv2
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
# from mcquic.validate.metrics import MsSSIM, PSNR
from mcquic.validate.handlers import MsSSIM, PSNR
from mcquic.utils.vision import DeTransform

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if not self.img_list[idx].endswith(".png") or self.img_list[idx].endswith(".jpg"):
            raise ValueError("Mistake format")
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = read_image(img_path, ImageReadMode.UNCHANGED)
        if self.transform:
            image = self.transform(image)
        return image

def load_model(model_path):
    print("load model...")
    compressor = Neon(channel=256, k=4096, size=[16, 8, 4, 2, 1], denseNorm=False)
    compressor.eval().cuda()
    
    print(f"load checkpoints from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    compressor.load_state_dict(
        {
            k[len("module._compressor.") :]: v
            for k, v in state_dict["trainer"]["_model"].items()
            if "_lpips" not in k
        }
    )
    for params in compressor.parameters():
        params.requires_grad_(False)

    return compressor


def main(args):
    data_path = os.path.join(args.root, args.dataset)
    # 1. load model
    ms_ssim = MsSSIM().to(0)
    psnr = PSNR().to(0)
    compressor = load_model(args.ckpt)
    # 2. load data
    eval_transform = T.Compose([
        T.ConvertImageDtype(torch.float32),
        AlignedCrop(256),
        T.Normalize(0.5, 0.5),
    ])
    detransform = DeTransform().to(0)
    dataset = CustomImageDataset(args.inp_path, transform=eval_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    # 3. inference
    print("inference data")
    psnr_res = []
    msssim_res = []
    img_restored = []
    # import ipdb
    # ipdb.set_trace()
    with torch.no_grad():
        for item in tqdm(dataloader):
            # print(item.shape)
            image = item.cuda()
            # to_pil_image(image.squeeze(0) * 255).save(f"./inp_1.png")
            codes, binaries, headers = compressor.compress(image)
            # print(len(codes))
            image_res = compressor.decompress(binaries, headers)
            image = detransform(image)
            image_res = detransform(image_res)
            # import ipdb
            # ipdb.set_trace()
            img_restored.append(to_pil_image(image_res.squeeze(0)))
            # img_restored.append(image_res.squeeze(0).detach().cpu().numpy())
            p_res = psnr.handle(images=image, restored=image_res)[0]
            # print(p_res)
            m_res = ms_ssim.handle(images=image, restored=image_res)[0]
            # print(m_res)
            psnr_res.append(p_res)
            msssim_res.append(m_res)
    
    # 4. calculate metrics
    mean_psnr = sum(psnr_res) / len(psnr_res)
    mean_msssim = sum(msssim_res) / len(msssim_res)

    print(f"PSNR: {mean_psnr}, MS-SSIM: {mean_msssim}")
    
    # 5. save results
    res_path = f"./results/eval/{args.dataset}"
    os.makedirs(res_path, exist_ok=True)
    for idx, item in enumerate(img_restored):
        # cv2_image = np.transpose(item, (1, 2, 0))
        # cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(os.path.join(res_path, f"{idx}.png"), cv2_image)
        item.save(os.path.join(res_path, f"{idx}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default="results/tokenizers/saved_mcq/val_20000.ckpt")
    parser.add_argument("--precision", default="fp32", choices=["bf16", "fp32"])
    parser.add_argument("--hf_token", type=str, default=None, help="huggingface read token for accessing gated repo.")
    parser.add_argument("--dataset", type=str, default="kodak", choice=["kodak", "clic2024"], help="huggingface read token for accessing gated repo.")
    parser.add_argument("--root", type=str, default="/ssdfs/datahome/tj24011/datasets/raw", help="infer data")
    
    args = parser.parse_known_args()[0]
    
    main(args)


    
    
    


