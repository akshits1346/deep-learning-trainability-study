import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    input_dir = "experiments/logs_noise"
    output_dir = "plots/noise_accuracy"
    os.makedirs(output_dir, exist_ok=True)

    for noise in [0, 20, 40]:
        data = np.load(f"{input_dir}/noise_{noise}_stats.npz")
        epochs = data["epoch"]
        train_acc = data["train_acc"]
        test_acc = data["test_acc"]

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, test_acc, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Label Noise = {noise}%")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/noise_{noise}_accuracy.png")
        plt.close()

    print("Saved label-noise accuracy plots.")


if __name__ == "__main__":
    main()

