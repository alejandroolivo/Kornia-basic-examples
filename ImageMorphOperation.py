import kornia as K
import torch

#custom imports
import Util.Plotting as Plotting
import Util.LoadNSave as Loading

# load image as torch tensor
img_t: torch.Tensor = Loading.load_image('./test images/tomates/tomates (1).jpg', print_tensor_shape=False)

# check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create kernel for morphological operations
kernel = torch.tensor([[0, 1, 1, 0],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [0, 1, 1, 0]]).to(device)

# perform morphological operations
dilated_image = K.morphology.dilation(img_t, kernel) # Dilation
eroded_image = K.morphology.erosion(img_t, kernel) # Erosion
open_image = K.morphology.opening(img_t, kernel) # Opening
close_image = K.morphology.closing(img_t, kernel) # Closing


# plot image with kornia
Plotting.plot_image(dilated_image, title='Dilated image')
Plotting.plot_image(eroded_image, title='Eroded image')
Plotting.plot_image(open_image, title='Opened image')
Plotting.plot_image(close_image, title='Closed image')