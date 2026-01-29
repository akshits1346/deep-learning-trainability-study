import torch
import numpy as np
import os
from collections import defaultdict

from models.mlp import MLP
from utils.training import train_one_epoch, evaluate
from utils.noisy_dataset import NoisyLabelsDataset
from utils.initialization import init_he

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def run_experiment(noise_percent):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 15
    batch_size = 128

    # ---------- Transforms ----------
    train_transform = transforms.ToTensor()
    test_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ])

    # ---------- Dataset ----------
    base_train = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transform
    )

    if noise_percent > 0:
        train_data = NoisyLabelsDataset(
            base_train,
            noise_fraction=noise_percent / 100.0,
            num_classes=10
        )
    else:
        train_data = base_train

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # ---------- Model ----------
    model = MLP(
        input_dim=784,
        hidden_dim=256,
        depth=8,
        num_classes=10
    ).to(device)

    init_he(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    grad_log = defaultdict(list)

    train_accs, test_accs = [], []

    # ---------- Training ----------
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_log
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(
            f"[Noise {noise_percent}%] "
            f"Epoch {epoch+1:02d} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Shifted Test Acc: {test_acc:.2f}%"
        )

    return train_accs, test_accs


def main():
    os.makedirs("experiments/logs_shift", exist_ok=True)

    for noise in [0, 40]:
        train_accs, test_accs = run_experiment(noise)

        np.save(
            f"experiments/logs_shift/shift_noise_{noise}.npy",
            np.array([train_accs, test_accs])
        )

    print("\nDataset shift experiments complete.")


if __name__ == "__main__":
    main()

