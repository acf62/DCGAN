import torch.nn as nn


# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
nfg = 64

# Size of feature maps in discriminator
nfd = 64


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, nfg * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nfg * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(nfg * 4, nfg * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfg * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(nfg * 2, nfg * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfg * 1),
            nn.ReLU(True),

            nn.ConvTranspose2d(nfg * 1, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, nfd, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfd, nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfd * 2, nfd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfd * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
