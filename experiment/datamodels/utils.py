import torch
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
from sklearn.metrics import silhouette_score

def resize_and_crop(img, size=(100,100), crop_type='middle'):
    # If height is higher we resize vertically, if not we resize horizontally
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    # The image is scaled/cropped vertically or horizontally
    # depending on the ratio
    if ratio > img_ratio:
        img = img.resize((
            size[0],
            int(round(size[0] * img.size[1] / img.size[0]))),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (
                0,
                int(round((img.size[1] - size[1]) / 2)),
                img.size[0],
                int(round((img.size[1] + size[1]) / 2)))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((
            int(round(size[1] * img.size[0] / img.size[1])),
            size[1]),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (
                int(round((img.size[0] - size[0]) / 2)),
                0,
                int(round((img.size[0] + size[0]) / 2)),
                img.size[1])
        elif crop_type == 'bottom':
            box = (
                img.size[0] - size[0],
                0,
                img.size[0],
                img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else:
        img = img.resize((
            size[0],
            size[1]),
            Image.ANTIALIAS)
    # If the scale is the same, we do not need to crop
    return img


def plot_sweeps_mnist(decoder, option, ax=None):
    # TODO: replace with ImageGrid (mpl_toolkits) and torchvision
    n = 15
    fig = np.zeros((28*n, 28*n))
    grid_x = np.linspace(-4, 4, n)
    grid_y = grid_x[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            if option == "cVAE_s":
                z_sample = np.array([[0, 0, xi, yi]])
                z_sample = torch.tensor(z_sample).to(torch.float32)
                x_decoded = decoder(z_sample)
                digit = x_decoded.reshape(28,28).detach().numpy()
                fig[i*28:(i+1)*28, j*28:(j+1)*28] = digit
            elif option == "cVAE_z":
                z_sample = np.array([[xi, yi, 0, 0]])
                z_sample = torch.tensor(z_sample).to(torch.float32)
                x_decoded = decoder(z_sample)
                digit = x_decoded.reshape(28,28).detach().numpy()
                fig[i*28:(i+1)*28, j*28:(j+1)*28] = digit
            elif option == "VAE":
                z_sample = np.array([[xi, yi]])
                z_sample = torch.tensor(z_sample).to(torch.float32)
                x_decoded = decoder(z_sample)
                digit = x_decoded.reshape(28,28).detach().numpy()
                fig[i*28:(i+1)*28, j*28:(j+1)*28] = digit
            else:
                print(f"unsupported option: {option}")    

    start_range = 28 // 2
    end_range = n * 28 + start_range + 1
    pixel_range = np.arange(start_range, end_range, 28)[:-1]
    
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)

    if ax is None:
        ax = plt.figure() 
    ax.set_xticks(pixel_range, sample_range_x)
    ax.set_yticks(pixel_range, sample_range_y)
    ax.set_xlabel("Latent Variable 1")
    ax.set_ylabel("Latent Variable 2")
    ax.set_title(option)
    ax.imshow(fig, cmap="Greys_r")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

def s_score(encoder, x, y):
    _, z_mean, _ = encoder(x)
    z_mean = z_mean.detach().numpy()
    ss = silhouette_score(z_mean, y.detach().numpy())
    return ss, z_mean

def plot_latent_space(encoder, x, y, ax=None):
    ss, z_mean = s_score(encoder, x, y)

    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(z_mean[:,0], z_mean[:,1], c=y, cmap="Accent")
    ax.set_title(f"Silhouette score: {ss:.4f}")
    ax.axis("off")