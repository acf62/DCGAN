import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


import DCGAN32 as DCGAN


# Number of training epochs
num_epochs = 1

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Initialize BCELoss function
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0


def initialize_weights(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


G = DCGAN.Generator(ngpu=ngpu).to(device)
D = DCGAN.Discriminator(ngpu=ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    G = nn.DataParallel(G, list(range(ngpu)))
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    D = nn.DataParallel(D, list(range(ngpu)))

G.apply(initialize_weights)
D.apply(initialize_weights)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))


# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, DCGAN.nz, 1, 1, device=device)


def train(data_loader):
    img_list = []
    iters = 0

    for epoch in range(num_epochs):
        for data in data_loader:
            D.zero_grad()

            batch = data[0].to(device)
            output = D(batch).view(-1)
            batch_size = batch.size(0)
            labels = torch.full((batch_size, ), real_label, device=device)
            err_d_real = criterion(output, labels)
            err_d_real.backward()

            noise = torch.randn(batch_size, DCGAN.nz, 1, 1, device=device)
            fake = G(noise)
            output = D(fake.detach()).view(-1)
            labels.fill_(fake_label)
            err_d_fake = criterion(output, labels)
            err_d_fake.backward()
            err_d = err_d_real + err_d_fake

            optimizerD.step()

            G.zero_grad()
            labels.fill_(real_label)
            output = D(fake).view(-1)
            err_g = criterion(output, labels)
            err_g.backward()

            optimizerG.step()

            if iters % 5 == 0:
                print(str(iters) + " out of " + str(len(data_loader)))
                with torch.no_grad():
                    fake = G(fixed_noise).detach().cpu()
                    save_image(vutils.make_grid(fake, padding=2, normalize=True), str(iters) + ".png")
            iters += 1
