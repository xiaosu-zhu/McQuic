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
from datasets import load_dataset
import matplotlib.pyplot as plt

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
        self.img_list = os.listdir(self.img_dir)[:4]

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
        # T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        # AlignedCrop(256),
        # T.Resize((256, 256)),
        T.RandomResizedCrop((256, 256)),
        T.Normalize(0.5, 0.5),
    ])
    def eval_trans(example):
        image = example['jpeg']
        return {"img_tensor": eval_transform(image)}

    def filter_incorrect(example):
        is_filter = False
        is_jpeg = False
        if example['jpeg'].mode != "L":
            is_jpeg = True
        if example['jpeg'].size[0] > 256 or example['jpeg'].size[1] > 256:
            is_filter = True
        return is_filter and is_jpeg
        
    detransform = DeTransform().to(0)
    dataset = CustomImageDataset(data_path, transform=eval_transform)
    # dataset = load_dataset(
    #     "webdataset", data_dir="/ssdfs/datahome/tj24011/datasets/raw/imagenet/imagenet-1k", split="validation", streaming=True
    # ).filter(filter_incorrect).map(eval_trans)
    # dataset = dataset.remove_columns("jpeg")
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    # 3. inference
    print("inference data")
    all_res_dis_v1 = []
    all_res_dis_v2 = []
    all_img_res = []
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        for idx, item in tqdm(enumerate(dataloader)):
            # print(item.shape)
            try:
                image = item.cuda()
                # image = item['img_tensor'].cuda()
                # to_pil_image(image.squeeze(0) * 255).save(f"./inp_1.png")
                # x = compressor._padding(image)
                # y = compressor._encoder(x)
                # yhat = compressor._decoder(y)
                # codes, binaries, headers = compressor.compress(image)
                # encoder
                x = compressor._encoder(image)
                # quantizer.encode
                allLatents = [] # large to small
                for encoder in compressor._quantizer._encoders:
                    x = encoder(x)
                    allLatents.append(x)
                _all_latents = [x.flatten(2).detach().cpu() / x.flatten(2).norm(p=2, dim=1).detach().cpu() for x in allLatents]

                codes = []
                currentLatent = torch.zeros_like(allLatents[-1])
                for quantizer, dequantizer, backward, latent in zip(compressor._quantizer._quantizers[::-1], compressor._quantizer._dequantizers[::-1], compressor._quantizer._backwards[::-1], allLatents[::-1]):
                    residual = latent - currentLatent
                    code = quantizer.encode(residual)
                    codes.append(code)
                    currentLatent = backward(latent)

                # visualization, visualize image per scale with accumulation
                # quantizer.decode
                intermedia_res = []
                intermedia_res_f1 = []
                intermedia_res_f1_wo = []
                formerLevel = None
                for idx, (decoder, dequantizer, code) in enumerate(zip(compressor._quantizer._decoders[::-1], compressor._quantizer._dequantizers[::-1], codes)):
                    quantized = dequantizer.decode(code)
                    if formerLevel is None:
                        intermedia_formerLevel = quantized.clone()
                        _f = quantized.clone()
                        import ipdb; ipdb.set_trace()
                        formerLevel = decoder(quantized) # for next scale
                    else:
                        intermedia_formerLevel = (quantized + formerLevel).clone()
                        _f = quantized + formerLevel
                        formerLevel = decoder(_f) # for next scale
                    
                    intermedia_f1 = _f.clone()
                    for _decoder in compressor._quantizer._decoders[-idx-2::-1]:
                        intermedia_f1 = _decoder(intermedia_f1)
                    intermedia_res.append(intermedia_formerLevel.flatten(2).detach().cpu() / intermedia_formerLevel.flatten(2).norm(p=2, dim=1).detach().cpu())
                    intermedia_res_f1.append(intermedia_f1.flatten(2).detach().cpu() / intermedia_f1.flatten(2).norm(p=2, dim=1).detach().cpu())
                    intermedia_res_f1_wo.append(intermedia_f1.detach())
                    
                # vis_v1: f1 - f1', f2 - f2', ...
                patch_dis_v1 = [] # 1, 2, 4, 8, 16
                for f, fhat in zip(_all_latents[::-1], intermedia_res):
                    # f_norm = np.linalg.norm(f)
                    # fhat_norm = np.linalg.norm(fhat)
                    # f = f / f_norm # (batchsize, feature_map, h, w)
                    # fhat = fhat / fhat_norm
                    patch_dis_v1.append((f - fhat).abs().mean())
                all_res_dis_v1.append(patch_dis_v1)
                
                # vis_v2: f5' - f1, f4' - f1, ....
                f1 = _all_latents[0]
                # f1_norm = np.linalg.norm(f1)
                # f1 = f1 / f1_norm
                patch_dis_v2 = []
                img_res = []
                for idx, (fhat, fhat_real) in enumerate(zip(intermedia_res_f1, intermedia_res_f1_wo)):
                    # fhat_norm = np.linalg.norm(fhat)
                    # fhat = fhat / fhat_norm
                    img = compressor._decoder(fhat_real)
                    img = detransform(img)
                    img = img.squeeze(0)
                    img_res.append(img.cpu())
                    patch_dis_v2.append((f1 - fhat).abs().mean())
                all_img_res.append(img_res)
                all_res_dis_v2.append(patch_dis_v2)
            except Exception as e:
                # torch.save(all_res_dis_v1, "./all_res_dis_v1.pth")
                # torch.save(all_res_dis_v2, "./all_res_dis_v2.pth")
                # print("allLatents", [x.shape for x in allLatents])
                print(e)
                continue
            # decoder    
            # for idx, res in enumerate(intermedia_res):
            #     img = compressor._decoder(res)
            #     pre_res = img
            #     img = detransform(img)
            #     to_pil_image(img.squeeze(0)).save(f"./{idx}.png")

            # image_res = compressor.decompress(binaries, headers)
            # image = detransform(image)
            # image_res = detransform(image_res)
            # # image_res = detransform(yhat)
            # img_restored.append(to_pil_image(image_res.squeeze(0)))
            # # img_restored.append(image_res.squeeze(0).detach().cpu().numpy())
            # p_res = psnr.handle(images=image, restored=image_res)[0]
            # # print(p_res)
            # m_res = ms_ssim.handle(images=image, restored=image_res)[0]
            # # print(m_res)
            # psnr_res.append(p_res)
            # msssim_res.append(m_res)
    
    # torch.save(all_res_dis_v1, "./all_res_dis_v1.pth")
    # torch.save(all_res_dis_v2, "./all_res_dis_v2.pth")
    plt.figure()
    for i in range(1, 5):
        for j in range(1, 6):
            plt.subplot(5,4, (j-1) * 4 + i)
            _img = to_pil_image(all_img_res[i - 1][j - 1])
            plt.imshow(_img)
            plt.xticks([])
            plt.yticks([])
    plt.savefig("./vis.png")
    # 4. calculate metrics
    res_dict_v1 = {"1": [], "2": [], "4": [], "8": [], "16": []}
    for batch in all_res_dis_v1:
        for pn, b in zip(list(res_dict_v1.keys()), batch):
            res_dict_v1[pn].append(b)
    
    res_dict_v2 = {"1": [], "2": [], "4": [], "8": [], "16": []}
    for batch in all_res_dis_v2:
        for pn, b in zip(list(res_dict_v2.keys()), batch):
            res_dict_v2[pn].append(b)
            
    for idx, (k, v) in zip(reversed(range(1, 6)), res_dict_v1.items()):
        print(f"pn = {k}: {sum(v) / len(v)}")
    for idx, (k, v) in zip(reversed(range(1, 6)), res_dict_v2.items()):
        print(f"pn = {k}: {sum(v) / len(v)}")

    # mean_psnr = sum(psnr_res) / len(psnr_res)
    # mean_msssim = sum(msssim_res) / len(msssim_res)

    # print(f"PSNR: {mean_psnr}, MS-SSIM: {mean_msssim}")
    
    # 5. save results
    # res_path = f"./results/eval/{args.dataset}_{args.steps}"
    # os.makedirs(res_path, exist_ok=True)
    # for idx, item in enumerate(img_restored):
    #     # cv2_image = np.transpose(item, (1, 2, 0))
    #     # cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    #     # cv2.imwrite(os.path.join(res_path, f"{idx}.png"), cv2_image)
    #     item.save(os.path.join(res_path, f"{idx}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--steps", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="results/tokenizers/saved_mcq/latest/val_200000.ckpt")
    parser.add_argument("--precision", default="fp32", choices=["bf16", "fp32"])
    parser.add_argument("--hf_token", type=str, default=None, help="huggingface read token for accessing gated repo.")
    parser.add_argument("--dataset", type=str, default="kodak", choices=["kodak", "clic2024"], help="huggingface read token for accessing gated repo.")
    parser.add_argument("--root", type=str, default="/ssdfs/datahome/tj24011/datasets/raw", help="infer data")
    
    args = parser.parse_known_args()[0]
    
    main(args)
