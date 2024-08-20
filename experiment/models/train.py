import torch
import matplotlib.pyplot as plt 

def batch_run_vae(model, x):
  # single batch run
  out = model(x)
  loss = model.overall_loss(x, out["x_hat"], out["z_mean"], out["z_logvar"])
  return loss

def batch_run_cvae(model, x):
  # single batch run
  tg, bg = x[:,0,:], x[:,1,:]
  out = model(tg, bg)
  loss = model.overall_loss(tg, out["tg_outputs"], bg, out["bg_outputs"], 
                            out["tg_z"], out["tg_z_mean"], out["tg_z_logvar"], 
                            out["tg_s"], out["tg_s_mean"], out["tg_s_logvar"],
                            out["bg_z"], out["bg_z_mean"], out["bg_z_logvar"])
  return loss

def train(model, dataloader, optimizer, epochs=50, plot=True):
  model.train()

  losses = []
  for epoch in range(epochs):
    for batch_idx, x in enumerate(dataloader):
      optimizer.zero_grad()
      if model.name == "VAE":
        loss = batch_run_vae(model, x)
      elif model.name == "cVAE":
        loss = batch_run_cvae(model, x)
      else:
        print(f"unsupported model: {model.name}")

      losses.append(loss)
      loss["overall_loss"].backward()
      optimizer.step()

  if plot:
    _, axes = plt.subplots(ncols=len(loss), figsize=(15,3)) 
    for ax, k in zip(axes, loss.keys()):
      ax.plot([l[k].detach().numpy() for l in losses])
      ax.set_title(k)

    # _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(ncols=5, figsize=(15,3))
    # ax1.plot([l[0].detach().numpy() for l in losses]); ax1.set_title("Reconstruction Loss")
    # ax2.plot([l[1].detach().numpy() for l in losses]); ax2.set_title("KL Loss")
    # ax3.plot([l[2].detach().numpy() for l in losses]); ax3.set_title("TC Loss")
    # ax4.plot([l[3].detach().numpy() for l in losses]); ax4.set_title("Discriminator Loss")
    # ax5.plot([l[4].detach().numpy() for l in losses]); ax5.set_title("Overall Loss")
  return losses