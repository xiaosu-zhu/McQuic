from mcquic.modules.generator import Generator
from PIL import Image
from mcquic.datasets.transforms import getTrainingPreprocess, getTrainingTransform, getEvalTransform
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
import torch

def test():
    generator = Generator(32, [1, 1, 1, 1, 1], [4096, 4096, 4096, 4096, 4096], loadFrom='compressor/compressor_19000.ckpt').cuda()
    # preprocess = getTrainingPreprocess()
    transform = getEvalTransform()
    # image = Image.open('valid/kodim01.png').convert('RGB')
    # [c, h, w]
    # image = to_tensor(image)
    image = torch.rand(2, 3, 512, 512)
    # [1, c, h, w]
    image = transform(image)
    print(image.shape)
    with torch.autocast('cuda', torch.bfloat16):
        predictions, codes, restored = generator(image.to('cuda'), ['Have a try', 'Another attempt with different size will test the variable length ability of the generator.'])

    print('****** PRES:', [pre.shape for pre in predictions], '********')
    print(sum([F.cross_entropy(pre, gt) for (pre, gt) in zip(predictions, codes)]))
    print(restored.shape)


if __name__ == '__main__':
    test()
