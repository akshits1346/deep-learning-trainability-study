import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    os.makedirs("plots/dataset_shift", exist_ok=True)

    for noise in [0, 40]:
        data = np.load(f"experiments/logs_shift/shift_noise_{noise}.npy")
        train_accs, test_accs = data

        plt.figure(figsize=(7, 4))
        plt.plot(train_accs, label="Train Accuracy")
        plt.plot(test_accs, label="Shifted Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Dataset Shift Performance — Noise {noise}%")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"plots/dataset_shift/shift_noise_{noise}.png"
        )
        plt.close()

    print("Saved dataset shift plots.")


if __name__ == "__main__":
    main()

