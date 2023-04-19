# Kornia-basic-examples

Kornia is an open-source computer vision library for PyTorch that provides various operations and augmentations on tensors, such as images and meshes. It is designed to be easy to use, efficient, and modular, making it suitable for research and industrial applications.

This repository contains a collection of basic computer vision operations and augmentations implemented using Kornia.

# Prerequisites

kornia==0.6.10

matplotlib==3.7.1

numpy==1.24.2

opencv-python==4.7.0.72

Pillow==9.4.0

torch==2.0.0

torchvision==0.15.1

# Installation

You can install Kornia via pip:

!pip install kornia

# Functions

load_image

This is a utility function that loads an image and converts it to a torch tensor. It takes a string data_path as input, which is the path to the image, and an optional boolean print_tensor_shape, which if set to True will print the shape of the tensor. The function returns a torch tensor of the image with values between 0 and 1.

load_image_batch

This function loads every image in a folder and returns a batch of images as a torch tensor. It takes a string data_path as input, which is the path to the folder containing the images, and an optional boolean print_tensor_shape, which if set to True will print the shape of the tensor. The function returns a torch tensor of the images with values between 0 and 1.

plot_image

This function takes a torch tensor of an image as input and plots the image using Matplotlib. It takes a torch tensor img as input, an optional string title as the title of the plot, and an optional integer nrows as the number of rows for the plot. The function plots the image and displays it.

RandomAffine

This function applies a random affine transformation to a batch of images. It takes a torch tensor img_batch_t as input and applies random rotation, translation, scaling, and shearing transformations to the images. The function returns the transformed batch of images as a torch tensor.

RandomCrop

This function applies a random crop to an image. It takes a torch tensor img_t as input and crops the image to a random size and location. The function returns the cropped image as a torch tensor.

# License

This project is licensed under the Apache License 2.0. For more information, please refer to the LICENSE file in the root of this repository.

Apache License 2.0
