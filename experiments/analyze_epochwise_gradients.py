import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    log_path = "experiments/logs_epochwise/epochwise_gradients.npz"
    output_dir = "plots/epochwise_gradients"
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(log_path)

    for layer_name in data.files:
        values = data[layer_name]
        epochs = values[:, 0]
        grads = values[:, 1]

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, grads, alpha=0.3)
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.title(f"Gradient Dynamics: {layer_name}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{layer_name}.png")
        plt.close()

    print("Saved epoch-wise gradient plots to:", output_dir)


if __name__ == "__main__":
    main()

