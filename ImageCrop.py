import kornia as K
import torch

#custom imports
import Util.Plotting as Plotting
import Util.LoadNSave as Loading

# load image as torch tensor
img_t: torch.Tensor = Loading.load_image('./test images/tomates/tomates (1).bmp', print_tensor_shape=True)

# create a random crop operation
crop_op = K.augmentation.AugmentationSequential(
    K.augmentation.RandomCrop((200, 200)),
    data_keys=["input"]
)

# apply the augmentation
img_t_cropped = crop_op(img_t)

# plot image with kornia
Plotting.plot_image(img_t_cropped, title='Image')