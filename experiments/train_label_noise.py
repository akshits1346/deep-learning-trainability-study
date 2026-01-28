import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import os

from models.mlp import MLP
from utils.training import set_seed
from utils.initialization import init_he
from utils.noisy_dataset import NoisyLabelsDataset


def run_for_noise(noise_frac, device):
    input_dim = 28 * 28
    hidden_dim = 256
    depth = 8
    num_classes = 10
    batch_size = 128
    epochs = 15
    lr = 1e-3

    transform = transforms.ToTensor()

    train_base = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_data = NoisyLabelsDataset(
        train_base, noise_fraction=noise_frac, num_classes=num_classes
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = MLP(input_dim, hidden_dim, depth, num_classes).to(device)
    init_he(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    stats = defaultdict(list)
    grad_logs = defaultdict(list)

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad_logs[name].append((epoch, p.grad.norm().item()))

            optimizer.step()

            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        train_acc = 100.0 * correct / total

        # ---- Test ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                correct += (out.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        test_acc = 100.0 * correct / total

        stats["epoch"].append(epoch)
        stats["train_acc"].append(train_acc)
        stats["test_acc"].append(test_acc)

        print(
            f"Noise={noise_frac} | Epoch {epoch+1}: "
            f"Train Acc={train_acc:.2f} | Test Acc={test_acc:.2f}"
        )

    return stats, grad_logs


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("experiments/logs_noise", exist_ok=True)

    for noise in [0.0, 0.2, 0.4]:
        print(f"\nRunning noise fraction = {noise}")
        stats, grads = run_for_noise(noise, device)

        np.savez(
            f"experiments/logs_noise/noise_{int(noise*100)}_stats.npz",
            **{k: np.array(v) for k, v in stats.items()}
        )
        np.savez(
            f"experiments/logs_noise/noise_{int(noise*100)}_grads.npz",
            **{k: np.array(v) for k, v in grads.items()}
        )


if __name__ == "__main__":
    main()

