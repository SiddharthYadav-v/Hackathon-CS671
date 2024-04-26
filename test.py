from model import *
from prepare_dataset import *
import matplotlib.pyplot as plt

PATH = './Diffusion_Model.pth'

model = DiffusionModel(T=4000, embed_size=64, n_channels=1, n_classes=1)
model.load_state_dict(torch.load(PATH))


def generate(model, batch_size=32):
    X = torch.randn((batch_size, 28, 28, 1))
    for t in range(T - 1, 0, -1):
        print(f'\rt = {t}', end=" ")
        noise = (torch.randn(X.shape)) if t > 1 else (torch.zeros(X.shape))
        X_noise = model(X, torch.LongTensor([t] * batch_size)).permute(0, 2, 3, 1)
        X = (
            1 / alpha[t] ** 0.5
            * (X - beta[t] / (1 - alpha_cumprod[t]) ** 0.5 * X_noise)
            + (1 - alpha[t]) ** 0.5 * noise
        )
    return X

X_gen = generate(model)

def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = images.squeeze(axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

plot_multiple_images(X_gen.numpy(), 8)
plt.show()
