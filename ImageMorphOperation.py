import kornia as K
import torch

#custom imports
import Util.Plotting as Plotting
import Util.LoadNSave as Loading

# load image as torch tensor
img_t: torch.Tensor = Loading.load_image('./test images/cells/cells (1).jpg', print_tensor_shape=False)

# define the rotation angle
angle: float = 17.36  # in degrees
angle = torch.tensor(angle)

# apply the transform to the image
img_out: torch.Tensor = K.geometry.rotate(img_t, angle)  # 1x3xHxW

# plot image with kornia
Plotting.plot_image(img_out, title='Rotated image')