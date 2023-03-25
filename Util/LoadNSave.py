import numpy as np
import kornia as K
import torch
import cv2
import os


def load_image(data_path: str, print_tensor_shape=False) -> torch.Tensor:
    """Utility function that load an image an convert to torch."""
    # open image using OpenCV (HxWxC)
    img: np.ndarray = cv2.imread(data_path, cv2.IMREAD_COLOR)

    # cast image to torch tensor and convert to RGB
    img_t: torch.Tensor = K.utils.image_to_tensor(img, keepdim=False)  # BxCxHxW
    img_t = K.color.bgr_to_rgb(img_t)

    if(print_tensor_shape):
        print(f"Tensor image shape: {img_t.shape}")

    return img_t.float() / 255.


def load_image_batch(data_path: str, print_tensor_shape=False) -> torch.Tensor:
    
    # load every image in the folder
    img_batch_t = torch.stack([load_image(data_path + '/' + img_name, print_tensor_shape) for img_name in os.listdir(data_path)])

    # Remove the extra dimension
    img_batch_t = img_batch_t.squeeze(1)
    
    # return the batch of images
    return img_batch_t
