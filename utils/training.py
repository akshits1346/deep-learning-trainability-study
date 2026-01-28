import torch
import numpy as np
from collections import defaultdict


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device, grad_log):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        # Gradient norm logging (raw)
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_log[name].append(p.grad.norm().item())

        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), 100.0 * correct / total

