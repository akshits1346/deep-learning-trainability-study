import numpy as np
import matplotlib.pyplot as plt
import os


def cosine_sim(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1).mean()


def compute_drift(reps, max_samples=2000, seed=42):
    rng = np.random.RandomState(seed)

    base = reps[0]
    n = base.shape[0]
    idx = rng.choice(n, size=min(max_samples, n), replace=False)

    base_sub = base[idx]
    drift = []

    for t in range(len(reps)):
        cur_sub = reps[t][idx]
        drift.append(cosine_sim(base_sub, cur_sub))

    return np.array(drift)


def main():
    print("SCRIPT STARTED")

    output_dir = "plots/representation_drift"
    os.makedirs(output_dir, exist_ok=True)

    for noise in [0, 40]:
        path = f"experiments/logs_repr/repr_noise_{noise}.npy"
        print(f"Loading {path}")

        reps = np.load(path)
        drift = compute_drift(reps)

        plt.figure(figsize=(8, 4))
        plt.plot(drift, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Cosine Similarity to Epoch 0")
        plt.title(f"Representation Drift — Noise {noise}%")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/drift_noise_{noise}.png")
        plt.close()

    print("Saved representation drift plots.")


if __name__ == "__main__":
    main()

