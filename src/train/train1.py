import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from src.data.loader import get_dl
from src.models.net import DermaNet
from src.utils.misc import seed, mkdir, pick_device
def one_epoch(model, dl, opt=None, dev="cpu"):
    model.train() if opt else model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    ys, ps = [], []
    for x, y in dl:
        if opt:
            x = x.to(dev)
            y = y.squeeze().long().to(dev)# medmnist labels often (N,1)
            opt.zero_grad()
        else:
            x = x.to(dev)
            y = y.squeeze().long().to(dev)

        logits = model(x)
        loss = loss_fn(logits, y)
        if opt:
            loss.backward()
            opt.step()
        losses.append(float(loss.detach().cpu()))
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        ps += pred
        ys += y.detach().cpu().numpy().tolist()
    acc = accuracy_score(ys, ps) if len(ys) else 0.0
    avg = sum(losses)/len(losses) if losses else 0.0
    return avg, acc
def main():
    seed(42)
    dev = pick_device()
    tr, va, te = get_dl(bs=128, download=True)
    model = DermaNet(n_classes=7).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mkdir("runs")
    best = -1
    # yeah just 6 epochs... (change if you want)
    for e in range(1,7):
        l1, a1 = one_epoch(model, tr, opt=opt, dev=dev)
        l2, a2 = one_epoch(model, va, opt=None, dev=dev)
        print("E", e, "train", l1, a1, "val", l2, a2)
        if a2 > best:
            best = a2
            torch.save(model.state_dict(), "runs/best.pt")
            with open("runs/last.txt","w", encoding="utf-8") as f:
                f.write("best_val_acc=" + str(best) + "\n")
    # quick random plot (not real history, sorry)
    plt.figure()
    plt.title("best val acc (yeah only 1 point)")
    plt.plot([0,1], [0, best])
    plt.savefig("runs/plot.png")
    plt.close()
if __name__ == "__main__":
    main()