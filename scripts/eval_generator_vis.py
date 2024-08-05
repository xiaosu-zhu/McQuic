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
    eval_transform = T.Compose([
        T.ConvertImageDtype(torch.float32),
        AlignedCrop(512),
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
    img_root = "/ssdfs/datahome/tj24011/workspace/McQuic/results/testset/imgs_test"
    img_list = sorted(os.listdir(img_root), key=lambda x: int(x.split(".")[0]))
    with open("/ssdfs/datahome/tj24011/workspace/McQuic/results/testset/t.txt", "r") as f:
        txt_list = f.readlines()

    compressor = generator.compressor
    text_encoder = generator.text_encoder
    tokenizer = generator.text_tokenizer
    transform = T.Compose([
        T.ToTensor(),
        # T.Resize(512),
        T.RandomResizedCrop((256, 256), (0.75, 1), (0.95, 1.05)),
        # T.ConvertImageDtype(torch.float32),
        RandomGamma()
    ])
    with torch.no_grad():
        for idx, (txt, img_path) in enumerate(tqdm(zip(txt_list, img_list))):
            image = Image.open(os.path.join(img_root, img_path))
            image = transform(image).cuda()
            image = image.unsqueeze(0)
            # image tokenize
            codes = compressor.encode(image.float())
            all_forwards_for_residual = list()
            formerLevel = None
            for level, code in enumerate(codes[:-1]):
                # list - 1 of [n, c, 2h, 2w]
                all_forwards_for_residual.append(
                    compressor.residual_forward(code, formerLevel, level)
                )
                formerLevel = all_forwards_for_residual[-1]

            # text tokenize
            batch_encoding = tokenizer(
                text=txt,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids = batch_encoding.input_ids.to(image.device)
            text_mask = batch_encoding.attention_mask.to(image.device)
    
            text_embedding = text_encoder(
                input_ids, attention_mask=text_mask, return_dict=True
            )
            
            # for i in range(5):
            #     new_all_forwards_for_residual = list()
            #     for x in all_forwards_for_residual[:-i]:
            #         n, c, h, w = x.shape
            #         x = x.permute(0, 2, 3, 1).reshape(n, h*w, -1)
            #         new_all_forwards_for_residual.append(x.to(torch.bfloat16))
            #     new_all_forwards_for_residual = torch.cat(new_all_forwards_for_residual, 1)

            #     last_hidden_state = text_embedding.last_hidden_state
            #     pooled_output = text_embedding.pooler_output
            #     attn_mask = torch.where(text_mask == 1, 0., -torch.inf)
                
            #     import ipdb; ipdb.set_trace()
            #     rawPredictions = generator.next_residual_predictor(
            #         last_hidden_state, pooled_output, attn_mask, new_all_forwards_for_residual
            #     )
                
            #     restoredCodes = [
            #         pre.detach().clone().argmax(1, keepdim=False) for pre in predictions
            #     ]
            #     restored = compressor.decode(restoredCodes)
                
            new_all_forwards_for_residual = list()
            for x in all_forwards_for_residual:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n, h*w, -1)
                new_all_forwards_for_residual.append(x.to(torch.bfloat16))
            new_all_forwards_for_residual = torch.cat(new_all_forwards_for_residual, 1)

            last_hidden_state = text_embedding.last_hidden_state
            pooled_output = text_embedding.pooler_output
            attn_mask = torch.where(text_mask == 1, 0., -torch.inf)

            rawPredictions = generator.next_residual_predictor(
                last_hidden_state, pooled_output, attn_mask, new_all_forwards_for_residual,
            )
            
            patch_nums = list(reversed(generator.size))
            curIdx = 0
            predictions = list()
            for pn in patch_nums:
                h = w = pn
                pre = rawPredictions[:, :, curIdx : curIdx + (h * w)]  # 1705
                pre = pre.permute(0, 3, 1, 2).reshape(1, -1, 4, h, w)
                predictions.append(pre)
                curIdx += h * w
                
            restoredCodes = [
                pre.detach().clone().argmax(1, keepdim=False) for pre in predictions
            ]
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                samples = compressor.decode(restoredCodes)
            
            img = detransform(samples)
            img = to_pil_image(img.squeeze(0))
            img.save(f"./{idx}.png")

            print(f"generated, cost: {dt * 1000}s")

    # 5. save results
    # res_path = f"./results/eval/generator/{args.dataset}"
    # os.makedirs(res_path, exist_ok=True)
    # for idx, item in enumerate(img_restored):
    #     # cv2_image = np.transpose(item, (1, 2, 0))
    #     # cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    #     # cv2.imwrite(os.path.join(res_path, f"{idx}.png"), cv2_image)
    #     item.save(os.path.join(res_path, f"{idx}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default="/ssdfs/datahome/tj24011/workspace/McQuic/results/generator/gen_mcq/latest/saved.ckpt")
    parser.add_argument("--tokenizer_path", type=str, default="results/tokenizers/saved_mcq/latest/val_200000.ckpt")
    parser.add_argument("--precision", default="fp32", choices=["bf16", "fp32"])
    parser.add_argument("--hf_token", type=str, default=None, help="huggingface read token for accessing gated repo.")
    parser.add_argument("--dataset", type=str, default="kodak", choices=["kodak", "clic2024"], help="huggingface read token for accessing gated repo.")
    parser.add_argument("--root", type=str, default="/ssdfs/datahome/tj24011/datasets/raw", help="infer data")
    
    args = parser.parse_known_args()[0]
    
    main(args)
