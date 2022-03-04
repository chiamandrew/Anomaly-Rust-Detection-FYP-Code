import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2

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
    img_array = Image.open(os.path.join(image_path,image_f)).convert('HSV')
    img_array = np.asarray(img_array)

    image_name = os.path.join(image_path, image_f)
    #print(os.path.join(image_path,image_f))

    #hsv = matplotlib.colors.rgb_to_hsv(img_array)
    #print(img_array)
    #plt.imshow(img_array)
    #plt.show()
    #raise
    #img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) #Should have been RGB not BGR
    #img_array = entropy(img_array,disk(5))
    #entropy_frame = (img_array * 255/np.max(img_array)).astype(np.uint8)
    
    saved_augmented_image_path = image_f.replace('.jpg', '.npy')
    saved_augmented_image_path = '/home/nwjbrandon/FYP/Pytorch-UNet/data/nbi/new_images_for_testing_(60-20-20)/hsv_imgs(800imgs)/' + saved_augmented_image_path
    #print(saved_augmented_image_path)
    ## Use these codes to Save Image
    #cv2.imwrite(saved_augmented_image_path, entropy_frame)
    np.save(saved_augmented_image_path, img_array)

    image_number += 1
    print(i)
