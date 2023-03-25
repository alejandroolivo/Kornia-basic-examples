import numpy as np
import matplotlib.pyplot as plt
import kornia as K
import torch, torchvision

# define method to plot image
def plot_image(img: torch.Tensor, title: str = 'Image', nrows=2) -> None:
    out: torch.Tensor = torchvision.utils.make_grid(img, nrows, padding=5)
    out_np: np.ndarray = K.utils.tensor_to_image(out)
    plt.title(title)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()
