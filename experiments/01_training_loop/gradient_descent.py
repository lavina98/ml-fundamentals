"""Day 1 warm-up: fit y = 3x + 2 three ways, then sweep the learning rate.

  A. Manual gradients   - derive dL/dw, dL/db by hand, update with plain tensor math
  B. Autograd           - let PyTorch compute gradients, update by hand
  C. nn.Module + optim  - the standard PyTorch training loop

Run:  python gradient_descent.py
Out:  loss_curves.png (loss vs step for each method, and for several learning rates)
"""
import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)

TRUE_W, TRUE_B = 3.0, 2.0
N, STEPS = 200, 100

x = torch.empty(N, 1).uniform_(-5, 5)
y = TRUE_W * x + TRUE_B + torch.randn(N, 1) * 0.5  # noisy line


def manual(lr):
    """A. No autograd. For L = mean((wx + b - y)^2):
    dL/dw = mean(2 * (wx + b - y) * x),  dL/db = mean(2 * (wx + b - y))"""
    w, b = torch.zeros(1), torch.zeros(1)
    losses = []
    for _ in range(STEPS):
        err = w * x + b - y
        losses.append((err ** 2).mean().item())
        w -= lr * (2 * err * x).mean()
        b -= lr * (2 * err).mean()
    return losses, w.item(), b.item()


def with_autograd(lr):
    """B. PyTorch builds the graph and computes gradients; we still do the update."""
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    losses = []
    for _ in range(STEPS):
        loss = ((w * x + b - y) ** 2).mean()
        losses.append(loss.item())
        loss.backward()              # fills w.grad, b.grad
        with torch.no_grad():        # the update itself must not be tracked
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()               # grads accumulate by default - reset them
        b.grad.zero_()
    return losses, w.item(), b.item()


def with_optim(lr):
    """C. The loop you'll write for every real model."""
    model = torch.nn.Linear(1, 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    losses = []
    for _ in range(STEPS):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, model.weight.item(), model.bias.item()


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    print(f"true: w={TRUE_W}, b={TRUE_B}")
    for name, fn in [("A manual", manual), ("B autograd", with_autograd), ("C optim", with_optim)]:
        losses, w, b = fn(lr=0.01)
        print(f"{name:<11} final loss={losses[-1]:.4f}  w={w:.3f}  b={b:.3f}")
        ax1.plot(losses, label=name)
    ax1.set(title="Same problem, three methods (lr=0.01)", xlabel="step", ylabel="MSE loss", yscale="log")
    ax1.legend()

    # For this data the loss curvature w.r.t. w is 2*E[x^2] ~= 16.7, so plain GD
    # diverges once lr > 2/16.7 ~= 0.12. Predict what 0.15 does before running.
    for lr in [0.001, 0.01, 0.1, 0.15]:
        losses, w, b = with_optim(lr)
        print(f"lr={lr:<6} final loss={losses[-1]:.4g}  w={w:.3g}  b={b:.3g}")
        ax2.plot(losses, label=f"lr={lr}")
    ax2.set(title="Learning-rate sweep", xlabel="step", ylabel="MSE loss", yscale="log")
    ax2.legend()

    fig.tight_layout()
    fig.savefig("loss_curves.png", dpi=120)
    print("saved loss_curves.png")


if __name__ == "__main__":
    main()
