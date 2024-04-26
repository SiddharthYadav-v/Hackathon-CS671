import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset

def variance_schedule(T, s=0.008, max_beta=0.999):
    t = np.arange(T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha = np.clip(f[1:] / f[:-1], 1 - max_beta, 1)
    alpha = np.append(1, alpha).astype(np.float32)  # add α₀ = 1
    beta = 1 - alpha
    alpha_cumprod = np.cumprod(alpha)
    return alpha, alpha_cumprod, beta  # αₜ , α̅ₜ , βₜ for t = 0 to T

np.random.seed(42)
T = 4000
alpha, alpha_cumprod, beta = variance_schedule(T)

def prepare_batch(X):
    X = X.unsqueeze(-1).float() * 2 - 1
    X_shape = X.shape
    t = torch.randint(1, T + 1, (X_shape[0],))
    alpha_cm = torch.tensor(alpha_cumprod[t])
    alpha_cm = alpha_cm.reshape([X_shape[0]] + [1] * (len(X_shape) - 1))
    noise = torch.randn(X_shape)
    X_noisy = alpha_cm ** 0.5 * X + (1 - alpha_cm) ** 0.5 * noise
    return {
        "X_noisy": X_noisy,
        "time": t,
    }, noise

def prepare_dataset(dataset, batch_size=64, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    mapped_loader = (
        (prepare_batch(batch) for batch in dataloader)
    )

    return mapped_loader
