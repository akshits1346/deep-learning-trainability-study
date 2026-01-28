import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    data = np.load("experiments/logs_width/width_generalization_gaps.npy")

    widths = data[:, 0]
    gaps = data[:, 1]

    os.makedirs("plots/width_scaling", exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(widths, gaps, marker="o")
    plt.xlabel("Hidden Layer Width")
    plt.ylabel("Mean Train–Test Accuracy Gap")
    plt.title("Generalization Gap vs Width (40% Label Noise)")
    plt.tight_layout()
    plt.savefig("plots/width_scaling/generalization_gap_vs_width.png")
    plt.close()

    print("Saved width scaling generalization plot.")


if __name__ == "__main__":
    main()

