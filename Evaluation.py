import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import Training

# Root directory for dataset
dataroot = "data"

# Number of workers for data_loader
workers = 0

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 32


# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dsets.CIFAR10(root='data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


# Create the data_loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)


Training.train(data_loader)
