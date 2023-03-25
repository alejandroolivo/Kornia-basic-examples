import kornia as K

#custom imports
import Util.Plotting as Plotting
import Util.LoadNSave as Loading

# load image batch as torch tensor
img_batch_t = Loading.load_image_batch('./test images/tomates', print_tensor_shape=False)

#Apply random affine transformation
img_batch_t = K.augmentation.RandomAffine(degrees=45, translate=(0.1, 0.4), scale=(0.6, 1.4), shear=25)(img_batch_t)

# plot image with kornia
Plotting.plot_image(img_batch_t, title='Image batch', nrows=8)