import torch
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel, pairwise_kernels

def standardize(x):
	return (x - x.mean(dim=0)) / x.std(dim=0)

def get_rbf_kernel(x, gamma=1):
  pairwise_dists = torch.cdist(x, x, p=2) ** 2 # TODO: produce numerical error for diag elements
  # pairwise_dists = torch.from_numpy(euclidean_distances(x, x)) ** 2
  return (-gamma * pairwise_dists).exp()

def compute_kernel(x, y=None, kernel="linear", **params):
  if kernel not in PAIRWISE_KERNEL_FUNCTIONS:
    raise ValueError(f"invalid kernel: {kernel}")

  return pairwise_kernels(x, y, metric=kernel, filter_params=True, **params)

def get_hsic(x, y, center=False, gamma=1):
  x, y = standardize(x), standardize(y)
  K_x = get_rbf_kernel(x, gamma=gamma)
  K_y = get_rbf_kernel(y, gamma=gamma)

  if center:
    # TODO: compare with KernelCenterer
    n_samples = len(x)
    H = torch.eye(n_samples) - (1 / n_samples) * torch.ones(n_samples, n_samples)
    K_x = H @ K_x @ H
    K_y = H @ K_y @ H

  hsic = torch.sum(K_x * K_y)
  return hsic, K_x, K_y

def get_cka(x, y, gamma=1):
  hsic, K_x, K_y = get_hsic(x, y, center=True, gamma=gamma)
  K_x_norm = torch.linalg.norm(K_x)
  K_y_norm = torch.linalg.norm(K_y)
  
  cka = hsic / (K_x_norm * K_y_norm)
  return cka