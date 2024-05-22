from mcquic.modules.generator_3_self_attn import GeneratorV3SelfAttention
from PIL import Image
from mcquic.data.transforms import (
    getTrainingPreprocess,
    getTrainingTransform,
    getEvalTransform,
)
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
import torch


def test():
    generator = GeneratorV3SelfAttention(
        256,
        4096,
        [16, 8, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1],
        denseNorm=False,
        qk_norm=True,
        norm_eps=1e-5,
        loadFrom="/ssdfs/datahome/tj24011/workspace/McQuic/saved/latest/val_30000.ckpt",
    ).eval().cuda()

    print(list(name for name, _ in generator.named_parameters() if "gamma" in name))
    with torch.no_grad():
        class_label = torch.tensor([138, 256]).cuda()
        predictions, restored = generator(None, class_label)

    print("****** PRES:", [pre.shape for pre in predictions], "********")
    print(restored.shape)
    restored_img = restored.detach().cpu().permute(0, 2, 3, 1).numpy().astype("uint8")
    # print(restored_img[0])
    
    im = Image.fromarray(restored_img[0])
    im.save("./restored_img.png")
    
    # print(sum([F.cross_entropy(pre, gt) for (pre, gt) in zip(predictions, codes)]))


if __name__ == "__main__":
    test()
