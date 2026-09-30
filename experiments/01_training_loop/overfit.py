"""Day 3, part 2: force overfitting, then fix it.

1,000 training images + a bigger network + 50 epochs = a model that memorizes.
Then the same run with dropout, with weight decay, and with both.

YOUR PART: build_big_model() - marked TODO.

Run:  python overfit.py
Out:  overfit.png, plus a results table
"""
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import evaluate, get_loaders, train_one_epoch

TRAIN_SUBSET = 1_000
EPOCHS = 50

# name -> (dropout p, weight decay)
CONFIGS = {
    "no regularization": (0.0, 0.0),
    "dropout 0.5": (0.5, 0.0),
    "weight decay 1.0": (0.0, 1.0),
    "both": (0.5, 1.0),
}


# ---------------------------------------------------------------- YOUR PART

def build_big_model(dropout=0.0):
    """TODO: return an nn.Sequential with two hidden layers of 512 and dropout after each ReLU:

        Flatten
        Linear(784, 512) -> ReLU -> Dropout(dropout)
        Linear(512, 512) -> ReLU -> Dropout(dropout)
        Linear(512, 10)

    Hint: nn.Dropout(p). With p=0.0 it does nothing, so one function covers every config.
    Before running: how many parameters is this? (~5x the Day 2 model - check your guess.)
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(dropout),

        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(dropout),

        nn.Linear(512, 10)
    )


# ---------------------------------------------------------------- boilerplate

def preload(loader, device, shuffle, batch_size):
    """Decode every image once and keep the whole set on the GPU as one tensor.
    (This is the fix for the Day 2 data-loading bottleneck - part 3 measures it.)"""
    xs, ys = zip(*loader)
    ds = TensorDataset(torch.cat(xs).to(device), torch.cat(ys).to(device))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def val_acc_in_train_mode(model, loader):
    """Deliberately WRONG evaluation: dropout left on. Shows why model.eval() matters."""
    model.train()
    correct = total = 0
    for xb, yb in loader:
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += yb.size(0)
    return correct / total


def run(name, dropout, weight_decay, train_loader, val_loader, device):
    torch.manual_seed(0)
    model = build_big_model(dropout).to(device)
    # AdamW = Adam with weight decay done properly (decoupled from the gradient).
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    hist = {"train": [], "val": [], "acc": []}
    for _ in range(EPOCHS):
        hist["train"].append(train_one_epoch(model, train_loader, loss_fn, opt, device))
        val_loss, acc = evaluate(model, val_loader, loss_fn, device)
        hist["val"].append(val_loss)
        hist["acc"].append(acc)

    best = min(range(EPOCHS), key=lambda i: hist["val"][i])
    hist["summary"] = {
        "final train loss": hist["train"][-1],
        "final val loss": hist["val"][-1],
        "best val loss (epoch)": f"{hist['val'][best]:.3f} ({best + 1})",
        "final val acc": hist["acc"][-1],
        "val acc, dropout left ON": val_acc_in_train_mode(model, val_loader) if dropout else None,
    }
    return hist


def plot(results, path="overfit.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    for ax, (name, h) in zip(axes, results.items()):
        ax.plot(h["train"], label="train")
        ax.plot(h["val"], label="val")
        ax.set(title=name, xlabel="epoch")
        ax.legend()
    axes[0].set_ylabel("cross-entropy loss")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = get_loaders(train_subset=TRAIN_SUBSET)
    train_loader = preload(train_loader, device, shuffle=True, batch_size=64)
    val_loader = preload(val_loader, device, shuffle=False, batch_size=1024)
    print(f"device: {device} | train images: {TRAIN_SUBSET} | val images: 10,000 | epochs: {EPOCHS}")
    print(f"parameters: {sum(p.numel() for p in build_big_model().parameters()):,}\n")

    results = {}
    for name, (p, wd) in CONFIGS.items():
        results[name] = run(name, p, wd, train_loader, val_loader, device)
        print(f"== {name}")
        for k, v in results[name]["summary"].items():
            if v is not None:
                print(f"   {k:<26} {v:.3f}" if isinstance(v, float) else f"   {k:<26} {v}")

    plot(results)
    print("\nsaved overfit.png")


if __name__ == "__main__":
    main()
