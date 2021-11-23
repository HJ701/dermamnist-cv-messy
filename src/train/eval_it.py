import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.data.loader import get_dl
from src.models.net import DermaNet
from src.utils.misc import pick_device
def main():
    dev = pick_device()
    te = get_dl(bs=256, download=True)[2]
    model = DermaNet(n_classes=7).to(dev)
    model.load_state_dict(torch.load("runs/best.pt", map_location=dev))
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in te:
            x = x.to(dev)
            y = y.squeeze().long()
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu()
            ys += y.tolist()
            ps += pred.tolist()
    print("conf mat:")
    print(confusion_matrix(ys, ps))
    print(classification_report(ys, ps))
if __name__ == "__main__":
    main()