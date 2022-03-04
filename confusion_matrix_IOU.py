import torch
import torchmetrics
from torchmetrics import JaccardIndex #IOU
from torchvision import transforms

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix

import cv2
import numpy as np
from PIL import Image

import os, os.path



true_mask_path = "/new_images_for_testing_(60-20-20)/extra_imgs_v3(200imgs)/"
pred_mask_path = "/new_images_for_testing_(60-20-20)/Predicted Images/HSVE/"
valid_images = [".jpg",".gif",".png",".tga"]

tm = []
pm = []

# Save the name of the image
tm = sorted(os.listdir(true_mask_path))
pm = sorted(os.listdir(pred_mask_path))

length = len(tm)
IOU_list = []
cm_all = np.array([[0,0],[0,0]])
image_number = 1

### Github
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(15,8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cmap='Reds'
        print("Normalized Confusion Matrix")
    else:
        cmap='Greens'
        print('Confusion Matrix Without Normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes, rotation=90)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
###

for i in range(length):
    tm_f = tm[i]
    pm_f = pm[i]

    true_mask = Image.open(r"/home/Desktop/FYP Code/new_images_for_testing_(60-20-20)/extra_imgs_v3(200imgs)/" + tm_f)

    true_mask = true_mask.convert('1')

    true_mask = np.asarray(true_mask)
    true_mask = true_mask.astype(int)

    true_mask[true_mask==False] = 0
    true_mask[true_mask==True] = 1

    pred_mask = Image.open(r"/home/Desktop/FYP Code/new_images_for_testing_(60-20-20)/Predicted Images/HSVE/" + pm_f)
    pred_mask = np.asarray(pred_mask)
    
    pred_mask[pred_mask>0] = 1
    pred = pred_mask.reshape(-1)
    true = true_mask.reshape(-1)
    cm = confusion_matrix(pred,true)
    cm_all += cm
    pred_mask = torch.tensor(pred_mask)
    true_mask = torch.tensor(true_mask)


    jaccard = JaccardIndex(num_classes=2)
    jaccard_score = jaccard(pred_mask, true_mask)
    jaccard_score = float("{0:.4f}".format(jaccard_score))
    IOU_list.append(jaccard_score)

print(IOU_list)
print(cm_all)
plot_confusion_matrix(cm_all, ['Background','Rust'], False, title = 'Unnormalised')
plot_confusion_matrix(cm_all, ['Background','Rust'], True, title = 'Normalised')

n, bins, patches = plt.hist(x=IOU_list, bins=10, color='#0504aa',
                            alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('IOU Value')
plt.ylabel('Frequency')
plt.title('IOU Histogram')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()
