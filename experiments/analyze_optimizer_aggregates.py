import numpy as np
import matplotlib.pyplot as plt
import os


def aggregate(values):
    epochs = np.unique(values[:, 0])
    means, vars_ = [], []
    for e in epochs:
        grads = values[values[:, 0] == e][:, 1]
        means.append(np.mean(grads))
        vars_.append(np.var(grads))
    return epochs, np.array(means), np.array(vars_)


def main():
    input_dir = "experiments/logs_opts"
    output_dir = "plots/opt_aggregates"
    os.makedirs(output_dir, exist_ok=True)

    for opt in ["sgd", "sgd_momentum", "adam"]:
        data = np.load(f"{input_dir}/opt_{opt}.npz")

        layers = [
            k for k in data.files
            if "feature_extractor.0.weight" in k or "classifier.weight" in k
        ]

        for layer in layers:
            values = data[layer]
            epochs, means, vars_ = aggregate(values)

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
            plt.title(f"{opt.upper()} — {layer}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{opt}_{layer}.png")
            plt.close()

    print("Saved optimizer aggregate plots.")


if __name__ == "__main__":
    main()

