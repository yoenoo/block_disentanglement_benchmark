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
    return mean + eps * torch.exp(logvar/2.) # log-var trick
  
  def forward(self, x):
    x = self.input_layer(x)
    z = F.relu(x)
    z_mean, z_logvar = self.z_mean_layer(z), self.z_logvar_layer(z)
    z = self.reparametrize(z_mean, z_logvar)
    return z, z_mean, z_logvar

class Decoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim, use_bias=False):
    super().__init__()
    self.latent_layer = nn.Linear(2*latent_dim, hidden_dim, bias=use_bias)
    self.output_layer = nn.Linear(hidden_dim, input_dim, bias=use_bias)

  def forward(self, x):
    x = self.latent_layer(x)
    x = F.relu(x)
    x = self.output_layer(x)
    return x

class Discriminator(nn.Module):
  def __init__(self, latent_dim):
    super().__init__()
    self.fc = nn.Linear(2*latent_dim, 1)
    self.sigmoid = nn.Sigmoid()
    
  def forward(self, x):
    x = self.fc(x)
    x = self.sigmoid(x)
    return x

class ContrastiveVAE(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim, use_bias=False, tc=False, gamma=1, beta=1):
    super().__init__()

    self.z_encoder = Encoder(input_dim, hidden_dim, latent_dim, use_bias=use_bias)
    self.s_encoder = Encoder(input_dim, hidden_dim, latent_dim, use_bias=use_bias)
    self.decoder = Decoder(input_dim, hidden_dim, latent_dim, use_bias=use_bias)
    self.discriminator = Discriminator(latent_dim)

    self.name = "cVAE"
    self.tc = tc
    self.gamma = gamma
    self.beta = beta

  def forward(self, tg, bg):
    """
    tg = target; bg = background
    """
    tg_z, tg_z_mean, tg_z_logvar = self.z_encoder(tg)
    tg_s, tg_s_mean, tg_s_logvar = self.s_encoder(tg)
    # bg_s, bg_s_mean, bg_s_logvar = self.s_encoder(bg)
    bg_z, bg_z_mean, bg_z_logvar = self.z_encoder(bg) ## TODO: needs to be z, not s?

    tg_outputs = self.decoder(torch.cat([tg_z, tg_s], dim=1))
    zeros = torch.zeros_like(tg_z)
    # bg_outputs = self.decoder(torch.cat([zeros, bg_s]))
    bg_outputs = self.decoder(torch.cat([bg_z, zeros], dim=1))
    # fg_outputs = self.decoder(torch.cat([tg_z, zeros]))
    fg_outputs = self.decoder(torch.cat([zeros, tg_s], dim=1))

    return {
      "tg_outputs": tg_outputs,
      "bg_outputs": bg_outputs,
      "fg_outputs": fg_outputs,
      "tg_z": tg_z,
      "tg_z_mean": tg_z_mean,
      "tg_z_logvar": tg_z_logvar,
      "tg_s": tg_s,
      "tg_s_mean": tg_s_mean,
      "tg_s_logvar": tg_s_logvar,
      "bg_z": bg_z,
      "bg_z_mean": bg_z_mean,
      "bg_z_logvar": bg_z_logvar,
    }

  def reconstruction_loss(self, inputs, outputs):
    rec_loss = torch.sum(F.mse_loss(inputs, outputs, reduction="none"), axis=1) # sum over pixels
    return rec_loss.mean()
      
  def kl_loss(self, z_mean, z_logvar):
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), axis=1)
    return kl_loss.mean()
        
  def tc_loss(self, tg_z, tg_s):
    # assert len(tg_z) % 2 == 0, f"invalid shape tg_z ({tg_z.shape})"
    z1, z2 = torch.chunk(tg_z, 2, dim=0)
    
    # assert len(tg_s) % 2 == 0, f"invalid shape tg_s ({tg_s.shape})"
    s1, s2 = torch.chunk(tg_s, 2, dim=0)

    q = torch.concat([
      torch.concat([s1, z1], axis=1),
      torch.concat([s2, z2], axis=1),
    ], axis=0)
    q_score = self.discriminator(q)
    
    q_bar = torch.concat([
      torch.concat([s1, z2], axis=1),
      torch.concat([z1, s2], axis=1),
    ], axis=0)
    q_bar_score = self.discriminator(q_bar)

    tc_loss = (q_score / (1 - q_score)).log().mean()
    discriminator_loss = (-q_score.log() - (1 - q_bar_score).log()).mean()
    return tc_loss, discriminator_loss
      
  def overall_loss(
    self, 
    tg_inputs, tg_outputs, 
    bg_inputs, bg_outputs, 
    tg_z, tg_z_mean, tg_z_logvar, 
    tg_s, tg_s_mean, tg_s_logvar, 
    bg_z, bg_z_mean, bg_z_logvar,
  ):
    reconst_loss_tg = self.reconstruction_loss(tg_inputs, tg_outputs) 
    reconst_loss_bg = self.reconstruction_loss(bg_inputs, bg_outputs)
    reconst_loss = reconst_loss_tg + reconst_loss_bg

    kl_loss_tg_z = self.kl_loss(tg_z_mean, tg_z_logvar)
    kl_loss_tg_s = self.kl_loss(tg_s_mean, tg_s_logvar)
    kl_loss_bg_z = self.kl_loss(bg_z_mean, bg_z_logvar)
    kl_loss = kl_loss_tg_z + kl_loss_tg_s + kl_loss_bg_z

    tc_loss = self.tc_loss(tg_z, tg_s)

    if self.tc:
      tc_loss, discriminator_loss = self.tc_loss(tg_z, tg_s)
    else:
      tc_loss, discriminator_loss = 0, 0
    
    overall_loss = reconst_loss + self.beta * kl_loss + self.gamma * tc_loss + discriminator_loss
    return {"reconstruction_loss": reconst_loss, "kl_loss": kl_loss, "tc_loss": tc_loss, "d_loss": discriminator_loss, "overall_loss": overall_loss}