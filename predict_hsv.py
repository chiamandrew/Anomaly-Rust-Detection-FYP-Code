import argparse
import logging
import os
import cv2

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.io import imread
from skimage import img_as_uint

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    print(scale_factor)
    net.eval()
    #img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img_hsv = np.asarray(full_img)
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_RGB2HSV)
    img_hsv = BasicDataset.preprocess(Image.fromarray(img_hsv), scale_factor, is_mask=False)
    #img = full_img.copy()
    img_hsv = np.asarray(img_hsv)
    #print(img_rgb.shape)
    #img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    #print(full_img)
    #raise

    img_texture = np.asarray(full_img)
    entropy_img = cv2.cvtColor(img_texture, cv2.COLOR_RGB2GRAY) #reading image as opencv. Opencv is reading images in BGR format. But, in Pillow, reading images at RGB
    entropy_img = entropy(entropy_img,disk(5))
    entropy_img = (entropy_img * 255/np.max(entropy_img)).astype(np.uint8)
    entropy_img = Image.fromarray(entropy_img)
    entropy_img = BasicDataset.preprocess_entropy_frame(entropy_img, scale_factor)
    entropy_img = np.array([entropy_img]) # Changes from (150,150) to (1,150,150), 2 dim to 3 dim
    
    print(img_hsv.shape)
    print(entropy_img.shape)
    
    #img = np.vstack([img_hsv, entropy_img])
    #img = np.vstack([img_hsv,])
    img = torch.from_numpy(img_hsv)
    print(img.shape)

    #img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0) #put datapoint into array so that 1 becomes the batch size
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad(): # prevent updating the weights because this is predicting
        print(img.shape)
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=2) #changed n_channels from 3 to 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        print(filename)
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
