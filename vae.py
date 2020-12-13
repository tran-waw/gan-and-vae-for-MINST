import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import numpy as np
import math 

os.makedirs("vae_images", exist_ok=True)

#hyper parameters
P_epoch=30
P_batch_size=128
P_learning_rate=0.0003
P_save_img_interval=200

if torch.cuda.is_available():
    cuda = True
    device = 'cuda'
else:
    cuda = False
    device = 'cpu'


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoderlayer = nn.Linear(28*28, 400)
        self.mean = nn.Linear(400, 20)
        self.logvar = nn.Linear(400, 20)
        self.decoder1 = nn.Linear(20, 400)
        self.decoder2 = nn.Linear(400, 28*28)

    def encoder(self, x):
        y = self.encoderlayer(x)
        y1 = F.relu(y)
        mean = self.mean(y1)
        logvar = self.logvar(y1)
        return mean, logvar

    def reparametrization(self, mu, logvar):
        y = torch.exp(0.5 * logvar)
        z = torch.randn(y.size()).to(device) * y + mu
        return z

    def decode(self, y):
        z = self.decoder1(y)
        z1 = F.relu(z)
        z2 = self.decoder2(z1)
        z3 = torch.sigmoid(z2)
        return z3

    def forward(self, x):
        mu, logvar = self.encoder(x)
        y = self.reparametrization(mu, logvar)
        z = self.decode(y)
        generated_imgs = z.view(z.size(0), 1, 28, 28)
        return generated_imgs, z, mu, logvar


def Lossfunc(generated, origin, mu, logvar):
    BCE_loss = nn.BCELoss(reduction='sum')
    gene_loss = BCE_loss(generated, origin)
    KLD = (-0.5) * torch.sum(1 + logvar - torch.exp(logvar) - (mu * mu))
    return gene_loss + KLD


vae = VAE()
if cuda:
    vae.cuda()  

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=P_batch_size, shuffle=True)

# use Adaptive moment estimation to replace normal gradient descent to get better results
optimizer = torch.optim.Adam(vae.parameters(), lr=P_learning_rate)


for epoch in range(P_epoch):
    for i, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        '''
        if cuda:
            imgs, labels = imgs.to('cuda'), labels.to('cuda')  
        else:
            imgs, labels = imgs.to('cpu'), labels.to('cpu')
        '''

        optimizer.zero_grad()
        original_imgs = torch.flatten(imgs, start_dim=1)
        
        generated_imgs, generated_imgs_unformat, mu, logvar = vae(original_imgs)
        vae_loss = Lossfunc(generated_imgs_unformat, original_imgs, mu, logvar)
        vae_loss.backward()
        optimizer.step()

        print("epoch %d - batch %d " % (epoch, i))

        # save image every 200 batches in epoch
        if i % P_save_img_interval == 0:
            save_image(generated_imgs.data[:30], "vae_images/%d-%d.png" % (epoch, i), nrow=10)