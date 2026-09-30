"""Day 3, part 1: inference with the checkpoint saved by train.py.

YOUR PART: predict() - marked TODO.

Run:  python infer.py          (after python train.py)
Out:  test accuracy, per-class accuracy, latency/throughput table, predictions.png
"""
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from train import CHECKPOINT, CLASSES, build_model


# ---------------------------------------------------------------- YOUR PART

def predict(model, xb, device):
    """TODO: return (predicted_class, confidence) for a batch of images.

      predicted_class: LongTensor (B,)  - index of the most likely class
      confidence:      FloatTensor (B,) - the probability the model gives that class (0..1)

    Steps:
      1. inference mode - which two things from Day 2 do you need, and why each?
      2. move xb to device, get logits
      3. probabilities = torch.softmax(logits, dim=1)   <- here we DO want softmax: we want real probabilities
      4. confidence, predicted_class = probabilities.max(dim=1)
      5. return both on the CPU (.cpu()) so the caller can compare with CPU labels / plot them
    """

    model.eval()
    xb = xb.to(device)
    with torch.no_grad():
        logits = model(xb)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = probabilities.max(dim=1)
        return predicted_class.to("cpu"), confidence.to("cpu")



# ---------------------------------------------------------------- boilerplate

def load_model(device):
    model = build_model()                                              # 1. same architecture as training
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)  # 2. just the numbers
    model.load_state_dict(state)                                       # 3. pour numbers into architecture
    return model.to(device)


def test_accuracy(model, loader, device):
    correct = torch.zeros(10)
    count = torch.zeros(10)
    for xb, yb in loader:
        pred, _ = predict(model, xb, device)
        for c in range(10):
            mask = yb == c
            correct[c] += (pred[mask] == c).sum()
            count[c] += mask.sum()
    return (correct.sum() / count.sum()).item(), (correct / count).tolist()


def time_batches(model, test_ds, device, batch_sizes=(1, 64, 1024), runs=50):
    """Median latency per batch and throughput. Inputs pre-built so we time the model, not the data."""
    print(f"\n{'batch':>6} | {'ms/batch':>9} | {'images/s':>10}")
    for bs in batch_sizes:
        xb = torch.stack([test_ds[i][0] for i in range(bs)])
        for _ in range(5):                        # warm-up: first calls pay one-time CUDA setup costs
            predict(model, xb, device)
        times = []
        for _ in range(runs):
            if device == "cuda":
                torch.cuda.synchronize()          # GPU runs async - wait for it before starting the clock
            t0 = time.perf_counter()
            predict(model, xb, device)            # .cpu() inside predict also forces a wait
            times.append(time.perf_counter() - t0)
        ms = sorted(times)[len(times) // 2] * 1000
        print(f"{bs:>6} | {ms:>9.3f} | {bs / ms * 1000:>10,.0f}")


def show_predictions(model, test_ds, device, n=10, path="predictions.png"):
    xb = torch.stack([test_ds[i][0] for i in range(n)])
    yb = [test_ds[i][1] for i in range(n)]
    pred, conf = predict(model, xb, device)
    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for ax, img, p, c, y in zip(axes.flat, xb, pred, conf, yb):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"{CLASSES[p]} ({c:.0%})\ntrue: {CLASSES[y]}",
                     color="green" if p == y else "red", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    # The TEST set: 10k images never used in training or validation.
    test_ds = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_ds, batch_size=256)

    acc, per_class = test_accuracy(model, test_loader, device)
    print(f"test accuracy: {acc:.2%}\n")
    for name, a in sorted(zip(CLASSES, per_class), key=lambda t: t[1]):
        print(f"  {name:<11} {a:.1%}")

    time_batches(model, test_ds, device)
    show_predictions(model, test_ds, device)
    print("\nsaved predictions.png")


if __name__ == "__main__":
    main()
