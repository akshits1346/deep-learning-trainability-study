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


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Config ----------------
    input_dim = 28 * 28
    hidden_dim = 256
    depth = 4
    num_classes = 10
    batch_size = 128
    epochs = 15
    lr = 1e-3

    os.makedirs("experiments/logs_epochwise", exist_ok=True)

    # ---------------- Data ----------------
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # ---------------- Model ----------------
    model = MLP(input_dim, hidden_dim, depth, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---------------- Epoch-wise gradient tracking ----------------
    epoch_gradients = defaultdict(list)

    for epoch in range(epochs):
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            for name, p in model.named_parameters():
                if p.grad is not None:
                    epoch_gradients[name].append(
                        (epoch, p.grad.norm().item())
                    )

            optimizer.step()

        print(f"Completed epoch {epoch + 1}/{epochs}")

    # ---------------- Save logs ----------------
    save_dict = {}
    for name, values in epoch_gradients.items():
        save_dict[name] = np.array(values)

    np.savez("experiments/logs_epochwise/epochwise_gradients.npz", **save_dict)


if __name__ == "__main__":
    main()

