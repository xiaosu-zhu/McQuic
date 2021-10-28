from tqdm import tqdm, trange
import torch
import torchvision
from PIL import Image

from compressai.zoo import cheng2020_attn, cheng2020_anchor, mbt2018, bmshj2018_factorized

evalStep = 5
testInput = 6

@torch.inference_mode()
def testSpeed():
    # [3, 768, 512]
    image = Image.open("data/kodak/kodim01.png").convert("RGB")
    image = torchvision.transforms.ToTensor()(image)
    image = image[None, ...].cuda().expand(testInput, *image.shape)
    model = mbt2018(quality=2, pretrained=True).cuda().eval()
    result = model.compress(image)
    binary, shape = result["strings"], result["shape"]

    xHat = model.decompress(binary, shape)

    torch.cuda.synchronize()

    startEvent = torch.cuda.Event(enable_timing=True)
    endEvent = torch.cuda.Event(enable_timing=True)

    startEvent.record()
    # test encoder
    for _ in trange(evalStep):
        model.compress(image)
    endEvent.record()
    torch.cuda.synchronize()
    encoderMs = startEvent.elapsed_time(endEvent) / (evalStep * testInput)

    startEvent = torch.cuda.Event(enable_timing=True)
    endEvent = torch.cuda.Event(enable_timing=True)

    startEvent.record()
    # test encoder
    for _ in trange(evalStep):
        model.decompress(binary, shape)
    endEvent.record()
    torch.cuda.synchronize()
    decoderMs = startEvent.elapsed_time(endEvent) / (evalStep * testInput)

    return {"encoderForwardTime": encoderMs, "decoderForwardTime": decoderMs}


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    print(testSpeed())
