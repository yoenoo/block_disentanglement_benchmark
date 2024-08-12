import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar
    
    def reparameterization(self, mean, var):
        eps = torch.randn_like(var)
        return mean + var * eps
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def calculate_loss(self, x, x_hat, mean, log_var, beta=1):
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reconstruction_loss + beta*kl_loss

def train_vae(model, optimizer, train_dataloader, epochs, verbose = 0):
    model.train()
    
    losses = []
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_dataloader):
          
            optimizer.zero_grad()
            
            x_hat, mean, logvar = model(x)
            loss = model.calculate_loss(x, x_hat, mean, logvar)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        epoch_loss = overall_loss / len(train_dataloader)
        losses.append(epoch_loss)

        if verbose > 0:
            print("\tEpoch", epoch, "\tAverage Loss: ", round(epoch_loss, 4))
        
    return losses


class ContrastiveVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, batch_size, disentangle=False, gamma=1, beta=1):
        super().__init__()

        self.disentangle = disentangle
        self.gamma = gamma
        self.beta = beta
        
        self.z_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
               
        self.z_mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar_layer = nn.Linear(hidden_dim, latent_dim)
    
        self.s_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.s_mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.s_logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(2*latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.D = nn.Sequential(
            nn.Linear(batch_size, 1),
            nn.Sigmoid(),
        )

    def reparameterization(self, mean, var):
        eps = torch.randn_like(var)
        return mean + var * eps
    
    def z_encode(self, x):
        x = self.z_encoder(x)
        mean, logvar = self.z_mean_layer(x), self.z_logvar_layer(x)
        z = self.reparameterization(mean, logvar)
        return z, mean, logvar

    def s_encode(self, x):
        x = self.s_encoder(x)
        mean, logvar = self.s_mean_layer(x), self.s_logvar_layer(x)
        s = self.reparameterization(mean, logvar)
        return s, mean, logvar

    def decode(self, x):
        return self.decoder(x)

    def forward(self, tg, bg):
        tg_z, tg_z_mean, tg_z_logvar = self.z_encode(tg)
        tg_s, tg_s_mean, tg_s_logvar = self.s_encode(tg)
        bg_s, bg_s_mean, bg_s_logvar = self.s_encode(bg)

        tg_outputs = self.decode(torch.cat((tg_z, tg_s), dim=-1))
        zeros = torch.zeros_like(tg_z)
        bg_outputs = self.decode(torch.cat((zeros, bg_s), dim=-1))
        fg_outputs = self.decode(torch.cat((tg_z, zeros), dim=-1))
        
        return tg_z, tg_z_mean, tg_z_logvar, tg_s, tg_s_mean, tg_s_logvar, bg_s_mean, bg_s_logvar, tg_outputs, bg_outputs, fg_outputs

    def calculate_loss(self, tg_inputs, bg_inputs, tg_outputs, bg_outputs, tg_z, tg_z_mean, tg_z_logvar, tg_s, tg_s_mean, tg_s_logvar, bg_s_mean, bg_s_logvar):
        reconstruction_loss_tg = F.mse_loss(tg_inputs, tg_outputs, reduction="sum")
        reconstruction_loss_bg = F.mse_loss(bg_inputs, bg_outputs, reduction="sum")
        reconstruction_loss = reconstruction_loss_tg + reconstruction_loss_bg
        
        kl_loss_tg_z = 1 + tg_z_logvar - tg_z_mean.pow(2) - tg_z_logvar.exp()
        kl_loss_tg_s = 1 + tg_s_logvar - tg_s_mean.pow(2) - tg_s_logvar.exp()
        kl_loss_bg_s = 1 + bg_s_logvar - bg_s_mean.pow(2) - bg_s_logvar.exp()
        kl_loss = -0.5 * torch.sum(kl_loss_tg_z + kl_loss_tg_s + kl_loss_bg_s)

        if self.disentangle:
            z1, z2 = torch.chunk(tg_z, 2, dim=0)
            s1, s2 = torch.chunk(tg_s, 2, dim=0)
            q_bar = torch.concat([
                torch.concat([s1, z2], axis=1),
                torch.concat([s2, z1], axis=1),
            ], axis=0).T
            q = torch.concat([
                torch.concat([s1, z1], axis=1),
                torch.concat([s2, z2], axis=1),
            ], axis=0).T
            q_bar_score = self.D(q_bar)
            q_score = self.D(q)
            tc_loss = (q_bar_score / (1 - q_score)).log()
            D_loss = -q_score.log() - (1-q_bar_score).log()
            cvae_loss = reconstruction_loss.mean() + self.beta * kl_loss.mean() + self.gamma * tc_loss.mean() + D_loss.mean()
        else:
            cvae_loss = reconstruction_loss.mean() + self.beta * kl_loss.mean()

        return cvae_loss

def train_cvae(model, optimizer, train_dataloader, epochs, verbose = 0):
    model.train()
    
    losses = []
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_dataloader):
            tg, bg = x[:,0,:], x[:,1,:]
            
            optimizer.zero_grad()
            
            tg_z, tg_z_mean, tg_z_logvar, tg_s, tg_s_mean, tg_s_logvar, bg_s_mean, bg_s_logvar, tg_outputs, bg_outputs, fg_outputs = model(tg, bg)
            
            loss = model.calculate_loss(tg, bg, tg_outputs, bg_outputs, tg_z, tg_z_mean, tg_z_logvar, tg_s, tg_s_mean, tg_s_logvar, bg_s_mean, bg_s_logvar)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            
        epoch_loss = overall_loss / len(train_dataloader)
        losses.append(epoch_loss)

        if verbose > 0:
            print("\tEpoch", epoch, "\tAverage Loss: ", round(epoch_loss, 4))
        
    return losses