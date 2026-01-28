import numpy as np
import matplotlib.pyplot as plt
import os


def aggregate(values):
    epochs = np.unique(values[:, 0])
    means = []
    for e in epochs:
        grads = values[values[:, 0] == e][:, 1]
        means.append(np.mean(grads))
    return epochs, np.array(means)


def main():
    input_dir = "experiments/logs_inits"
    output_dir = "plots/init_aggregates"
    os.makedirs(output_dir, exist_ok=True)

    for init in ["xavier", "he", "orthogonal"]:
        data = np.load(f"{input_dir}/init_{init}.npz")

        layers = [
            k for k in data.files
            if "feature_extractor.0.weight" in k or "classifier.weight" in k
        ]

        for layer in layers:
            values = data[layer]
            epochs, means = aggregate(values)

            plt.figure(figsize=(8, 4))
            plt.plot(epochs, means)
            plt.xlabel("Epoch")
            plt.ylabel("Mean Gradient Norm")
            plt.title(f"{init.capitalize()} Init — {layer}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{init}_{layer}.png")
            plt.close()

    print("Saved initialization aggregate plots.")
    

if __name__ == "__main__":
    main()

