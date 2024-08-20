import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim, use_bias=False):
    super().__init__()
    self.input_layer = nn.Linear(input_dim, hidden_dim, bias=use_bias)
    self.z_mean_layer = nn.Linear(hidden_dim, latent_dim, bias=use_bias)
    self.z_logvar_layer = nn.Linear(hidden_dim, latent_dim, bias=use_bias)
        
  def reparametrize(self, mean, logvar):
    eps = torch.randn_like(logvar)
    return mean + eps * torch.exp(logvar/2.)

  def forward(self, x):
    x = self.input_layer(x)
    z = F.relu(x)
    z_mean, z_logvar = self.z_mean_layer(z), self.z_logvar_layer(z)
    z = self.reparametrize(z_mean, z_logvar)
    return z, z_mean, z_logvar

class Decoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim, use_bias=False):
    super().__init__()
    self.latent_layer = nn.Linear(latent_dim, hidden_dim, bias=use_bias)
    self.output_layer = nn.Linear(hidden_dim, input_dim, bias=use_bias)

  def forward(self, x):
    x = self.latent_layer(x)
    x = F.relu(x)
    x = self.output_layer(x)
    return x

class VAE(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim, use_bias=False, beta=1):
    super().__init__()
    self.encoder = Encoder(input_dim, hidden_dim, latent_dim, use_bias=use_bias)
    self.decoder = Decoder(input_dim, hidden_dim, latent_dim, use_bias=use_bias)
    self.beta = beta
    self.name = "VAE"
    
  def forward(self, x):
    z, z_mean, z_logvar = self.encoder(x)
    x_hat = self.decoder(z)
    return {"z": z, 
            "z_mean": z_mean, 
            "z_logvar": z_logvar, 
            "x_hat": x_hat}

  def reconstruction_loss(self, inputs, outputs):
    rec_loss = torch.sum(F.mse_loss(inputs, outputs, reduction="none"), axis=1) # sum over pixels
    return rec_loss.mean()
      
  def kl_loss(self, z_mean, z_logvar):
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), axis=1)
    return kl_loss.mean()

  def overall_loss(self, x, xhat, z_mean, z_logvar):
    reconstruction_loss = self.reconstruction_loss(x, xhat)
    kl_loss = self.kl_loss(z_mean, z_logvar)
    overall_loss = reconstruction_loss + self.beta * kl_loss
    return {"reconstruction_loss": reconstruction_loss, 
            "kl_loss": kl_loss, 
            "overall_loss": overall_loss}