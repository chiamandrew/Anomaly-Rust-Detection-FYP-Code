import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.io import imread
from skimage import img_as_uint

from PIL import Image
import os, os.path
import random
from numpy.lib.type_check import _imag_dispatcher
import io

# random.seed(0)
# rng = np.random.default_rng()

# noise_mask = np.full((128, 128), 28, dtype=np.uint8)
# noise_mask[32:-32, 32:-32] = 30

# noise = (noise_mask * rng.random(noise_mask.shape) - 0.5
#          * noise_mask).astype(np.uint8)
# img = noise + 128

# entr_img = entropy(img, disk(10))

# fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

# img0 = ax0.imshow(noise_mask, cmap='gray')
# ax0.set_title("Object")
# ax1.imshow(img, cmap='gray')
# ax1.set_title("Noisy image")
# ax2.imshow(entr_img, cmap='viridis')
# ax2.set_title("Local entropy")

# fig.tight_layout()

#image = img_as_ubyte(data.camera())


# Names
image_name = 'image'
transformed_name1 = 'transformed'
transformed_mask1 = 'mask'

## Placing the images in trial folder into an array
imgs = []

image_path = "/home/nwjbrandon/FYP/Pytorch-UNet/data/nbi/new_images_for_testing_(60-20-20)/resized_imgs(800imgs)"
valid_images = [".jpg",".gif",".png",".tga"]

# Save the name of the image
imgs = sorted(os.listdir(image_path))
#print(sorted(os.listdir(image_path)))


length = len(imgs)
output_array = []

image_number = 1

for i in range(length):
    image_f = imgs[i]
    img_array = Image.open(os.path.join(image_path,image_f))
    img_array = np.asarray(img_array)

    image_name = os.path.join(image_path, image_f)
    print(os.path.join(image_path,image_f))

    #fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4),
    #                           sharex=True, sharey=True)

    #img0 = plt.imshow(img_array, cmap=plt.cm.gray)
    #ax0.set_title("Image")
    #ax0.axis("off")
    #fig.colorbar(img0, ax=ax0)

    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) #Should have been RGB not BGR
    img_array = entropy(img_array,disk(5))
    entropy_frame = (img_array * 255/np.max(img_array)).astype(np.uint8)
    
    #img1 = plt.imshow(entropy(img_array, disk(5)), cmap='gray')
    #ax1.set_title("Entropy")
    #ax1.axis("off")
    #fig.colorbar(img1, ax=ax1)

    #fig.tight_layout()

    #plt.show()
    #print(entropy_frame)
    #raise
    saved_augmented_image_path = image_f.replace('.jpg', '_Entropy.jpg')
    saved_augmented_image_path = '/home/nwjbrandon/FYP/Pytorch-UNet/data/nbi/new_images_for_testing_(60-20-20)/Entropy Images/' + saved_augmented_image_path
    print(saved_augmented_image_path)
    ## Use these codes to Save Image
    cv2.imwrite(saved_augmented_image_path, entropy_frame)
    #np.save(saved_augmented_image_path, entropy_frame)

    image_number += 1
    print(i)

#plot_examples(output_array) 
#plot_examples_mask(output_array_mask) # mask appears inverted

#image0 = np.array(masks[5], dtype=np.float64) # after changing type the masks work but not on plot_examples
#cv2.imshow(image_name, image0) 
#cv2.waitKey(5000)

## Saving the New Images


