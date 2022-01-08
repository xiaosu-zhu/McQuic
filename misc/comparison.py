from tqdm import tqdm, trange
import torch
import torchvision
from PIL import Image

from compressai.zoo import cheng2020_attn, cheng2020_anchor, mbt2018, bmshj2018_factorized

from mcqc.nn.convs import conv3x3, conv5x5


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


@torch.inference_mode()
def conv5vsConv3():
    evalStep = 500
    conv3 = conv3x3(192, 192, 1).cuda()
    conv5 = conv5x5(192, 192, 1).cuda()

    a = torch.randn(1, 192, 128, 128).expand(128, 192, 128, 128).cuda()

    torch.cuda.synchronize()

    startEvent = torch.cuda.Event(enable_timing=True)
    endEvent = torch.cuda.Event(enable_timing=True)

    for _ in trange(10):
        conv3(a)
    for _ in trange(10):
        conv5(a)

    startEvent.record()
    # test encoder
    for _ in trange(evalStep):
        conv3(a)
    endEvent.record()
    torch.cuda.synchronize()
    conv3Ms = startEvent.elapsed_time(endEvent) / (evalStep * len(a))

    startEvent.record()
    # test encoder
    for _ in trange(evalStep):
        conv5(a)
    endEvent.record()
    torch.cuda.synchronize()
    conv5Ms = startEvent.elapsed_time(endEvent) / (evalStep * len(a))

    return {"conv3ForwardTime": conv3Ms, "conv5ForwardTime": conv5Ms}


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    print(conv5vsConv3())
    # print(testSpeed())
