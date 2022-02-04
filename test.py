import torch

def num2bit(x: torch.ByteTensor):
    x = x.cpu().numpy()
    bits = x.view('<f4')
    return torch.from_numpy(bits)

def bit2num(x: torch.FloatTensor):
    x = x.cpu().numpy()
    num = x.view('<u1')
    return torch.from_numpy(num)

if __name__ == "__main__":
    a = torch.randint(256, size=[1,4,8,256], dtype=torch.uint8)
    bits = num2bit(a)
    print(bits.shape)
    b = bit2num(bits)
    assert torch.all(a == b)
