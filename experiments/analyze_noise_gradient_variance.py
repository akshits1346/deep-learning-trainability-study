import numpy as np
import matplotlib.pyplot as plt
import os


def aggregate_variance(values):
    epochs = np.unique(values[:, 0])
    variances = []
    for e in epochs:
        grads = values[values[:, 0] == e][:, 1]
        variances.append(np.var(grads))
    return epochs, np.array(variances)


def main():
    output_dir = "plots/noise_gradient_variance"
    os.makedirs(output_dir, exist_ok=True)

    for noise in [0, 40]:
        data = np.load(f"experiments/logs_noise/noise_{noise}_grads.npz")

        # Early layer = most sensitive
        key = "feature_extractor.0.weight"
        values = data[key]

        epochs, var = aggregate_variance(values)

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, var, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Variance")
        plt.title(f"Early-layer Gradient Variance — Noise {noise}%")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gradvar_noise_{noise}.png")
        plt.close()

    print("Saved gradient variance plots.")


if __name__ == "__main__":
    main()

