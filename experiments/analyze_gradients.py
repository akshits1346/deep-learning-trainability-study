import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    log_path = "experiments/logs/gradient_logs.npz"
    output_dir = "plots/gradients"
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(log_path)

    layer_names = list(data.keys())

    # ---------------- Gradient Norm Statistics ----------------
    means = {}
    stds = {}

    for layer in layer_names:
        grads = data[layer]
        means[layer] = np.mean(grads)
        stds[layer] = np.std(grads)

    # ---------------- Plot: Mean Gradient Norms ----------------
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(layer_names)), means.values())
    plt.xticks(range(len(layer_names)), layer_names, rotation=90)
    plt.ylabel("Mean Gradient Norm")
    plt.title("Mean Gradient Norm per Layer")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mean_gradient_norms.png")
    plt.close()

    # ---------------- Plot: Gradient Variance ----------------
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(layer_names)), stds.values())
    plt.xticks(range(len(layer_names)), layer_names, rotation=90)
    plt.ylabel("Gradient Std Dev")
    plt.title("Gradient Variance per Layer")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gradient_variance.png")
    plt.close()

    print("Saved plots to:", output_dir)


if __name__ == "__main__":
    main()

