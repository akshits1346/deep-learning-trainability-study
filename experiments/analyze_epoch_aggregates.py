import numpy as np
import matplotlib.pyplot as plt
import os


def aggregate_per_epoch(values):
    epochs = np.unique(values[:, 0])
    means, vars_ = [], []
    for e in epochs:
        grads = values[values[:, 0] == e][:, 1]
        means.append(np.mean(grads))
        vars_.append(np.var(grads))
    return epochs, np.array(means), np.array(vars_)


def main():
    input_dir = "experiments/logs_depths"
    output_dir = "plots/epoch_aggregates"
    os.makedirs(output_dir, exist_ok=True)

    for depth in [2, 4, 8]:
        data = np.load(f"{input_dir}/depth_{depth}.npz")

        # Focus on representative layers
        layers_of_interest = [
            k for k in data.files
            if "feature_extractor.0.weight" in k or "classifier.weight" in k
        ]

        for layer in layers_of_interest:
            values = data[layer]
            epochs, means, vars_ = aggregate_per_epoch(values)

            plt.figure(figsize=(8, 4))
            plt.plot(epochs, means, label="Mean")
            plt.fill_between(
                epochs,
                means - np.sqrt(vars_),
                means + np.sqrt(vars_),
                alpha=0.3,
                label="Std"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Gradient Norm")
            plt.title(f"Depth {depth} — {layer}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/depth_{depth}_{layer}.png")
            plt.close()

    print("Saved aggregated epoch-wise plots.")


if __name__ == "__main__":
    main()

