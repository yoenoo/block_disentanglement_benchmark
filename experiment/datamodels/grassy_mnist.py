import numpy as np
import matplotlib.pyplot as plt 
np.random.seed(0)

import torch
import torchvision.datasets as datasets

from PIL import Image
from pathlib import Path
from .utils import resize_and_crop

# Imagenet data -> grass
IMAGENET_PATH = "./data/imagenet/grass_images/"

def get_grassy_mnist(scale):

  # data preparation
  mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=None)
  images, labels = mnist_train.data, mnist_train.targets

  # only the digits 0, 1, and 2 were used in the target dataset
  target_idx = np.where(labels < 3)[0] 
  foreground = images[target_idx,:].reshape(-1, 28*28) / 255. 
  target_labels = labels[target_idx]

  # IMAGENET -> grass images 
  grass_imgs = []
  for fpath in Path(IMAGENET_PATH).iterdir():
    if fpath.suffix != ".JPEG": 
      print("unknown file extension:", fpath)
      continue

    try:
      im = Image.open(fpath)
      im = im.convert(mode="L") # grayscale
      im = resize_and_crop(im, size=(100,100), crop_type="middle") # resize and crop to 100px x 100px
      grass_imgs.append(np.reshape(im, 10000))
    except Exception as e:
      print(e)
      print("unknown error:", fpath)
      
  grass_imgs = np.asarray(grass_imgs, dtype="float32") / 255. # rescale to 0-1
  grass_imgs = torch.tensor(grass_imgs)

  # pick random patches of grass images 
  rand_idxs = np.random.permutation(grass_imgs.shape[0])
  split = int(len(rand_idxs)) // 2

  target_idxs = rand_idxs[:split]
  background_idxs = rand_idxs[split:]

  target = torch.zeros_like(foreground)
  background = torch.zeros_like(foreground)

  for i in range(target.shape[0]):
    idx = np.random.choice(target_idxs) # randomly pick an image
    loc = np.random.randint(70, size=(2))
    grass_img = np.reshape(grass_imgs[idx,:], (100,100))
    superimposed_patch = np.reshape(grass_img[loc[0]:loc[0]+28,:][:,loc[1]:loc[1]+28],(1,784)) # randomly pick a region in the grass image
    target[i] = foreground[i] + scale * superimposed_patch
    
    idx = np.random.choice(background_idxs)
    loc = np.random.randint(70, size=(2))
    grass_img = np.reshape(grass_imgs[idx,:], (100,100))
    background_patch = np.reshape(grass_img[loc[0]:loc[0]+28,:][:,loc[1]:loc[1]+28],(1,784))
    background[i] = background_patch

  # normalize
  target /= target.max()
  background /= background.max()

  return target, target_labels, background