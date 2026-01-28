import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import os

from models.mlp import MLP
from utils.training import set_seed, train_one_epoch, evaluate


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Config ----------------
    input_dim = 28 * 28
    hidden_dim = 256
    depth = 4
    num_classes = 10
    batch_size = 128
    epochs = 10
    lr = 1e-3

    os.makedirs("experiments/logs", exist_ok=True)

    # ---------------- Data ----------------
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
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

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # ---------------- Model ----------------
    model = MLP(input_dim, hidden_dim, depth, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    grad_log = defaultdict(list)

    # ---------------- Training ----------------
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_log
        )

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%"
        )

    # ---------------- Save gradients ----------------
    grad_log = {k: np.array(v) for k, v in grad_log.items()}
    np.savez("experiments/logs/gradient_logs.npz", **grad_log)


if __name__ == "__main__":
    main()

