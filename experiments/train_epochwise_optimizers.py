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


def run_for_optimizer(opt_name, optimizer_fn, device):
    input_dim = 28 * 28
    hidden_dim = 256
    depth = 8
    num_classes = 10
    batch_size = 128
    epochs = 15
    lr = 1e-3

    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = MLP(input_dim, hidden_dim, depth, num_classes).to(device)
    init_he(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters(), lr)

    epochwise = defaultdict(list)

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
                    epochwise[name].append((epoch, p.grad.norm().item()))

            optimizer.step()

    return epochwise


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("experiments/logs_opts", exist_ok=True)

    optimizers = {
        "sgd": lambda params, lr: optim.SGD(params, lr=lr),
        "sgd_momentum": lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9),
        "adam": lambda params, lr: optim.Adam(params, lr=lr)
    }

    for name, opt_fn in optimizers.items():
        print(f"Running optimizer={name}")
        logs = run_for_optimizer(name, opt_fn, device)
        save_dict = {k: np.array(v) for k, v in logs.items()}
        np.savez(f"experiments/logs_opts/opt_{name}.npz", **save_dict)


if __name__ == "__main__":
    main()

