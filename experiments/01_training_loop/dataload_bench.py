"""Day 3, part 3: test the Day 2 theory - is the GPU waiting on data loading?

All mine to run - your job is to predict the numbers first, then explain them.

  A. Iterate the DataLoader alone (no model) with num_workers = 0, 2, 4
  B. One full training epoch: standard loader vs whole dataset preloaded on the device
  C. CPU vs GPU, both with preloaded data (so only the compute differs)

Run:  python dataload_bench.py     (takes ~1-2 min)
"""
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import build_model, get_loaders, train_one_epoch


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def one_epoch(loader, device):
    torch.manual_seed(0)
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return timed(lambda: train_one_epoch(model, loader, nn.CrossEntropyLoss(), opt, device))


def main():
    train_loader, _ = get_loaders()
    dataset = train_loader.dataset

    print("A. DataLoader alone, 50k images, no model")
    for workers in (0, 2, 4):
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=workers)
        print(f"   num_workers={workers}: {timed(lambda: [None for _ in loader]):.2f}s")

    print("\nB. Full epoch on GPU (data + forward + backward + step)")
    xs, ys = zip(*DataLoader(dataset, batch_size=1024))         # decode all images once
    x_all, y_all = torch.cat(xs), torch.cat(ys)
    print(f"   whole train set as one tensor: {x_all.nbytes / 1e6:.0f} MB")
    print(f"   standard loader:    {one_epoch(train_loader, 'cuda'):.2f}s")
    gpu_loader = DataLoader(TensorDataset(x_all.cuda(), y_all.cuda()), batch_size=64, shuffle=True)
    print(f"   preloaded on GPU:   {one_epoch(gpu_loader, 'cuda'):.2f}s")

    print("\nC. CPU vs GPU, preloaded data (compute only)")
    cpu_loader = DataLoader(TensorDataset(x_all, y_all), batch_size=64, shuffle=True)
    print(f"   CPU, batch 64:      {one_epoch(cpu_loader, 'cpu'):.2f}s")
    print(f"   GPU, batch 64:      {one_epoch(gpu_loader, 'cuda'):.2f}s")
    big_cpu = DataLoader(TensorDataset(x_all, y_all), batch_size=1024, shuffle=True)
    big_gpu = DataLoader(TensorDataset(x_all.cuda(), y_all.cuda()), batch_size=1024, shuffle=True)
    print(f"   CPU, batch 1024:    {one_epoch(big_cpu, 'cpu'):.2f}s")
    print(f"   GPU, batch 1024:    {one_epoch(big_gpu, 'cuda'):.2f}s")


if __name__ == "__main__":  # required on Windows: num_workers>0 re-imports this file in each worker
    main()
