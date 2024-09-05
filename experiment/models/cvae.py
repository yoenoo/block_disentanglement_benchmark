import torch
import torch.nn as nn
import torch.nn.functional as F
from .regularizer import get_hsic, get_cka

class MLPEncoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim, use_bias=False):
    super().__init__()
    self.input_layer = nn.Linear(input_dim, hidden_dim, bias=use_bias)
    self.z_m_layer = nn.Linear(hidden_dim, latent_dim, bias=use_bias)
    self.z_lv_layer = nn.Linear(hidden_dim, latent_dim, bias=use_bias)

  def reparametrize(self, m, lv):
    eps = torch.randn_like(lv)
    return m + eps * torch.exp(lv/2.) # log-var trick
  
  def forward(self, x):
    x = self.input_layer(x)
    z = F.relu(x)
    z_m, z_lv = self.z_m_layer(z), self.z_lv_layer(z)
    z = self.reparametrize(z_m, z_lv)
    return z, z_m, z_lv

class MLPDecoder(nn.Module):
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

class ConvEncoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim, use_bias=False, **conv_kwargs):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim, bias=use_bias, **conv_kwargs)
    self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=2*hidden_dim, bias=use_bias, **conv_kwargs)

    # TODO: can take 4*hidden_dim or through function argument
    self.h_layer = nn.Linear(2*hidden_dim, 4*hidden_dim, bias=use_bias)
    self.z_m_layer = nn.Linear(4*hidden_dim, latent_dim, bias=use_bias)
    self.z_lv_layer = nn.Linear(4*hidden_dim, latent_dim, bias=use_bias)
    
  def reparametrize(self, m, lv):
    eps = torch.randn_like(lv)
    return m + eps * torch.exp(lv/2.)
    
  def forward(self, x):
    z = self.conv1(x)
    z = F.relu(z)
    
    z = self.conv2(z)
    z = F.relu(z)

    z = z.permute(0,2,3,1) # reorder axes
    z = self.h_layer(z)
    z = F.relu(z)
    
    z_m, z_lv = self.z_m_layer(z), self.z_lv_layer(z)
    z = self.reparametrize(z_m, z_lv)
    return z, z_m, z_lv

class ConvDecoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim, use_bias=False, **conv_kwargs):
    super().__init__()
    self.h_layer = nn.Linear(2*latent_dim, 4*hidden_dim, bias=use_bias)
    self.conv1_t = nn.ConvTranspose2d(in_channels=4*hidden_dim, out_channels=2*hidden_dim, bias=use_bias, 
                                      output_padding=1, **conv_kwargs)
    self.conv2_t = nn.ConvTranspose2d(in_channels=2*hidden_dim, out_channels=hidden_dim, bias=use_bias, 
                                      output_padding=1, **conv_kwargs)
    
    ks = conv_kwargs["kernel_size"]
    assert ks % 2 == 1, f"requires odd kernel size, instead got {ks}"
    padding = int((ks-1) / 2) # i.e. "same"
    self.conv_out_t = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1, bias=use_bias, 
                                         kernel_size=ks, padding=padding)

  def forward(self, x):
    x = self.h_layer(x)
    
    x = x.permute(0,3,1,2) # reorder axes
    x = self.conv1_t(x)
    x = F.relu(x)

    x = self.conv2_t(x)
    x = F.relu(x)

    x = self.conv_out_t(x)
    # x = nn.Sigmoid(x) # TODO
    return x

class ContrastiveVAE(nn.Module):
  def __init__(self, Encoder, Decoder, input_dim, hidden_dim, latent_dim, use_bias=False, gamma=1, beta=1, regularizer=None, penalty=1, **kwargs):
    super().__init__()

    self.z_encoder = Encoder(input_dim, hidden_dim, latent_dim, use_bias=use_bias, **kwargs)
    self.s_encoder = Encoder(input_dim, hidden_dim, latent_dim, use_bias=use_bias, **kwargs)
    self.decoder = Decoder(input_dim, hidden_dim, latent_dim, use_bias=use_bias, **kwargs)
    self.discriminator = Discriminator(latent_dim)

    self.name = "cVAE"
    self.gamma = gamma
    self.beta = beta
    self.regularizer = regularizer
    self.penalty = penalty

  def forward(self, tg, bg):
    """tg = target; bg = background"""
    tg_z, tg_z_m, tg_z_lv = self.z_encoder(tg)
    tg_s, tg_s_m, tg_s_lv = self.s_encoder(tg)
    bg_z, bg_z_m, bg_z_lv = self.z_encoder(bg)

    tg_outputs = self.decoder(torch.cat([tg_z, tg_s], dim=-1))
    zeros = torch.zeros_like(tg_z)
    bg_outputs = self.decoder(torch.cat([bg_z, zeros], dim=-1)) # bg means s = 0
    
    # NOTE: fg_outputs not used in model training... for model inspection purpose...
    # fg_outputs = self.decoder(torch.cat([tg_z, zeros], dim=1)) -- from the official code
    fg_outputs = self.decoder(torch.cat([zeros, tg_s], dim=-1))

    return dict(
      tg_outputs=tg_outputs, bg_outputs=bg_outputs, fg_outputs=fg_outputs,
      tg_z=tg_z, tg_z_m=tg_z_m, tg_z_lv=tg_z_lv,
      tg_s=tg_s, tg_s_m=tg_s_m, tg_s_lv=tg_s_lv,
      bg_z=bg_z, bg_z_m=bg_z_m, bg_z_lv=bg_z_lv,
    )

  def reconstruction_loss(self, inputs, outputs):
    rec_loss = torch.sum(F.mse_loss(inputs, outputs, reduction="none"), axis=1) # sum over pixels
    return rec_loss.mean()
      
  def kl_loss(self, z_m, z_lv):
    kl_loss = -0.5 * torch.sum(1 + z_lv - z_m.pow(2) - z_lv.exp(), axis=1)
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
      
  def wasserstein_loss(self, z_m, z_lv):
    z_std = torch.exp(z_lv/2)
    w_loss = torch.norm(z_m, dim=-1).pow(2) + torch.norm(z_std, dim=-1).pow(2)
    return w_loss.mean()
      
  # TODO: staticmethod or in the utils
  def _rbf_kernel(self, X, sigma=1.0):
    dist = torch.cdist(X, X, p=2).pow(2)
    return torch.div(-dist, 2*sigma**2).exp()
      
  def HSIC_loss(self, s, z):
    # this doesn't work
    return get_hsic(s, z, center=True, gamma=self.gamma)[0]

  def CKA_loss(self, s, z):
    return get_cka(s, z, gamma=self.gamma)
      
  # def HSIC_loss(self, s_m, z_m):
  #   import sys; sys.path.append("../pysim")
  #   from pysim.pysim.kernel.hsic import HSIC

  #   hsic_clf = HSIC(center=True, kernel="rbf", gamma_X=1e-6, gamma_Y=1e-6)
  #   s_m = standardize(s_m)
  #   z_m = standardize(z_m)
  #   hsic_clf.fit(s_m.detach().numpy(), z_m.detach().numpy())
  #   hsic_score = hsic_clf.hsic_value
  #   # cka_score = hsic_clf.score(s_m.detach().numpy(), normalize=True)
  #   # cka_score = hsic_clf.score(s_m.detach().numpy(), normalize=False)
  #   # return torch.tensor(cka_score)
  #   return torch.tensor(hsic_score)

  # def HSIC_loss(self, s_m, z_m):
  #   n = s_m.shape[0]
  #   s_m = standardize(s_m)
  #   z_m = standardize(z_m)
  #   K = self._rbf_kernel(s_m, sigma=3.)
  #   L = self._rbf_kernel(z_m, sigma=3.)

  #   t1 = 1 / n**2 * torch.sum(K*L)
  #   t2 = 1 / n**4 * torch.sum(K) * torch.sum(L)
  #   t3 = 2 / n**3 * torch.sum(K@L)
  #   return t1 + t2 - t3

  def MMD_loss(self, s_m, z_m):
    K = self._rbf_kernel(s_m)
    L = self._rbf_kernel(z_m)
    return torch.norm(torch.mean(K) - torch.mean(L)).pow(2)
      
  def overall_loss(self, 
    tg_inputs, tg_outputs, 
    bg_inputs, bg_outputs, 
    tg_z, tg_z_m, tg_z_lv, 
    tg_s, tg_s_m, tg_s_lv, 
    bg_z, bg_z_m, bg_z_lv,
  ):
    reconst_loss_tg = self.reconstruction_loss(tg_inputs, tg_outputs) 
    reconst_loss_bg = self.reconstruction_loss(bg_inputs, bg_outputs)
    reconst_loss = reconst_loss_tg + reconst_loss_bg

    kl_loss_tg_z = self.kl_loss(tg_z_m, tg_z_lv)
    kl_loss_tg_s = self.kl_loss(tg_s_m, tg_s_lv)
    kl_loss_bg_z = self.kl_loss(bg_z_m, bg_z_lv) # -> bg_output
    kl_loss = kl_loss_tg_z + kl_loss_tg_s + kl_loss_bg_z

    if self.regularizer == "TC":
      tc_loss, discriminator_loss = self.tc_loss(tg_z, tg_s)
      reg_loss = tc_loss + discriminator_loss
    elif self.regularizer == "Wasserstein":
      w_loss = self.wasserstein_loss(tg_s_m, tg_s_lv) 
      # w_loss = self.wasserstein_loss(bg_s_m, bg_s_lv) # -- from original code
      reg_loss = w_loss
    elif self.regularizer == "HSIC":
      # hsic_loss = self.CKA_loss(tg_s_m, tg_z_m) + self.CKA_loss(tg_s_lv, tg_z_lv)
      # hsic_loss = self.HSIC_loss(tg_s, tg_z)
      hsic_loss = self.CKA_loss(tg_s, tg_z)
      reg_loss = hsic_loss
    elif self.regularizer == "MMD":
      mmd_loss = self.MMD_loss(tg_s_m, tg_z_m)
      reg_loss = mmd_loss
    elif self.regularizer is None:
      reg_loss = torch.tensor(0)
    else:
      raise ValueError(f"invalid regularizer: {self.regularizer}")

    overall_loss = reconst_loss + self.beta * kl_loss + self.penalty * reg_loss
    return dict(reconstruction_loss=reconst_loss, 
                kl_loss_tg_z=kl_loss_tg_z,
                kl_loss_tg_s=kl_loss_tg_s,
                kl_loss_bg_z=kl_loss_bg_z,
                kl_loss=kl_loss, 
                reg_loss=reg_loss, 
                overall_loss=overall_loss)