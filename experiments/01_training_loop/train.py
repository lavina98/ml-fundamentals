"""Day 2: train a small MLP on FashionMNIST.

Run:  python train.py
Out:  checkpoints/mlp.pt, train_val_loss.png
"""
import os
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
CHECKPOINT = "checkpoints/mlp.pt"

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# ---------------------------------------------------------------- boilerplate

def get_loaders(batch_size=BATCH_SIZE, train_subset=None):
    """50k train / 10k val split of the FashionMNIST training set.
    train_subset: use only the first N training samples (for Day 3's overfitting run)."""
    full = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    train_ds, val_ds = random_split(full, [50_000, 10_000], generator=torch.Generator().manual_seed(0))
    if train_subset:
        train_ds = torch.utils.data.Subset(train_ds, range(train_subset))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)  # no shuffle needed; bigger batch is fine without grads
    return train_loader, val_loader


def plot_curves(train_losses, val_losses, path="train_val_loss.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_losses, marker="o", label="train")
    plt.plot(epochs, val_losses, marker="o", label="val")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title("FashionMNIST MLP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ---------------------------------------------------------------- model + loops

def build_model():
    """Flatten -> Linear(784, 256) -> ReLU -> Linear(256, 10). 203,530 params.
    infer.py imports this to rebuild the same architecture before loading weights."""
    return nn.Sequential(
        nn.Flatten(),          # (B, 1, 28, 28) -> (B, 784)
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),    # one logit per class
    )


def train_one_epoch(model, loader, loss_fn, opt, device):
    """One pass over the training data. Returns the average training loss."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, loss_fn, device):
    """One pass without training. Returns (average loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item()
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)
    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------- driver

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, val_loader = get_loaders()
    model = build_model().to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, loss_fn, opt, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.perf_counter() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"acc {val_acc:.2%} | {elapsed:.1f}s")

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    plot_curves(train_losses, val_losses)
    print(f"saved {CHECKPOINT} and train_val_loss.png")


if __name__ == "__main__":
    main()
