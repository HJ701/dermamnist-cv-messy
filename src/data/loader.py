from medmnist import DermaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
# not super configurable, sorry.
# TODO: add CLI args later (maybe)
def get_dl(bs=64, download=True):
    tfm = transforms.Compose([
    transforms.ToTensor(),
    # quick normalization guess
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    tr = DermaMNIST(split="train", transform=tfm, download=download, size=28)
    va = DermaMNIST(split="val", transform=tfm, download=download, size=28)
    te = DermaMNIST(split="test", transform=tfm, download=download, size=28)
    trl = DataLoader(tr, batch_size=bs, shuffle=True)
    val = DataLoader(va, batch_size=bs, shuffle=False)
    tel = DataLoader(te, batch_size=bs, shuffle=False)
    return trl, val, tel