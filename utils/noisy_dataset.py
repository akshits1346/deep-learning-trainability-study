import torch
import numpy as np
from torch.utils.data import Dataset


class NoisyLabelsDataset(Dataset):
    def __init__(self, base_dataset, noise_fraction, num_classes=10, seed=42):
        self.base_dataset = base_dataset
        self.noise_fraction = noise_fraction
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)

        self.targets = np.array(base_dataset.targets)
        self.noisy_targets = self.targets.copy()

        n_noisy = int(len(self.targets) * noise_fraction)
        noisy_indices = self.rng.choice(len(self.targets), n_noisy, replace=False)

        for idx in noisy_indices:
            original = self.noisy_targets[idx]
            new_label = self.rng.randint(0, num_classes)
            while new_label == original:
                new_label = self.rng.randint(0, num_classes)
            self.noisy_targets[idx] = new_label

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        y = int(self.noisy_targets[idx])
        return x, y

