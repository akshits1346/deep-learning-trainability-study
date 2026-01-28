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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Experiment config ----------------
    widths = [64, 128, 256, 512]
    depth = 8
    noise_percent = 40
    epochs = 15
    batch_size = 128
    lr = 1e-3

    os.makedirs("experiments/logs_width", exist_ok=True)

    # ---------------- Dataset ----------------
    transform = transforms.ToTensor()

    base_train = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    train_data = NoisyLabelsDataset(
        base_train,
        noise_fraction=noise_percent / 100.0,
        num_classes=10
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    results = []

    # ---------------- Width sweep ----------------
    for width in widths:
        print(f"\nTraining width = {width}")

        model = MLP(
            input_dim=784,
            hidden_dim=width,
            depth=depth,
            num_classes=10
        ).to(device)

        init_he(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        train_accs = []
        test_accs = []

        # Dummy gradient log (required by train_one_epoch API)
        grad_log = defaultdict(list)

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                grad_log
            )

            test_loss, test_acc = evaluate(
                model,
                test_loader,
                criterion,
                device
            )

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            print(
                f"Epoch {epoch+1:02d}/{epochs} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}%"
            )

        gap = np.mean(np.array(train_accs) - np.array(test_accs))
        results.append((width, gap))

        np.save(
            f"experiments/logs_width/width_{width}_accs.npy",
            np.array([train_accs, test_accs])
        )

    np.save(
        "experiments/logs_width/width_generalization_gaps.npy",
        np.array(results)
    )

    print("\nWidth scaling experiments complete.")


if __name__ == "__main__":
    main()

