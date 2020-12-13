import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image 
from torch.autograd import Variable

import os
import numpy as np
import math  

os.makedirs("gan_images", exist_ok=True)   # the dir to store generated picture

#hyper parameters
P_epoch=200
P_batch_size=64
P_learning_rate=0.0002
P_save_img_interval=400

if torch.cuda.is_available():
    cuda = True
else:
    cuda = False

if cuda:
    Tensor = torch.cuda.FloatTensor  
else:
    Tensor = torch.FloatTensor

 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
 
        def Normalized_fullconnect_LeakyReLU(input1, output1, normalize=True):              # a combination of full connection, normalization and leakyrelu
            layers = [nn.Linear(input1, output1)]
            if normalize:
                layers.append(nn.BatchNorm1d(output1))           #to improve perfermance, avoid radient disappearance problem, add a normalization layer here, use the default parameters
            layers.append(nn.LeakyReLU(0.2, inplace=True))       #use LeakyReLU to replace Sigmoid to get better result. use inplace to save memory usage
            return layers
 
        self.model = nn.Sequential(
            *Normalized_fullconnect_LeakyReLU(100, 128, normalize=False),
            *Normalized_fullconnect_LeakyReLU(128, 256),
            *Normalized_fullconnect_LeakyReLU(256, 512),
            *Normalized_fullconnect_LeakyReLU(512, 1024),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
 
    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), 1, 28, 28)
        return img
 
 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
 
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),  #use LeakyReLU to replace Sigmoid to get better result. use inplace to save memory usage
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
 
    def forward(self, img):
        imgvectors = img.view(img.size(0), -1)
        discrimination = self.model(imgvectors)
        return discrimination
 
 
# use bce loss to replace vanilla loss of minimax algorithm
Lossfunc = torch.nn.BCELoss()
  
#initialization 
generator = Generator()
discriminator = Discriminator() 
if cuda:
    generator.cuda()
    discriminator.cuda()
    Lossfunc.cuda()
 
  
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=P_batch_size, shuffle=True)

 
 
# use Adaptive moment estimation to replace normal gradient descent to get better results
# If using parameter betas=(0.5, 0.999), there will be a incrediable improvement, almost reduce epochs needed 
# from 200 to 20. like paper:Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
# don't know why.(not explaned in paper too)
gene_gd = torch.optim.Adam(generator.parameters(), lr=P_learning_rate)            
dis_gd = torch.optim.Adam(discriminator.parameters(), lr=P_learning_rate)
 

 
for epoch in range(P_epoch):
    for i, (imgs, labels) in enumerate(dataloader):
        True_result = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)       #set a all 1s tensor that represents a picture, that be totally thought as a real number picture
        False_result = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)        #set a all 0s tensor that represents a picture, that be totally not thought as a real number picture
        original_imgs = Variable(imgs.type(Tensor))    # use later to get discriminator loss

        gene_gd.zero_grad()   # initialize gradiant descent optimizer
        dis_gd.zero_grad()   # initialize gradiant descent optimizer

        x = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 100))))
        generated_imgs = generator(x)
        g_loss = Lossfunc(discriminator(generated_imgs), True_result) 
        g_loss.backward()       # compute derivatives
        gene_gd.step()          # do adam gradient descent

        #the mean of loss of: 1.discriminate a real picture, loss to all true 2.discriminate a generated picture, loss to all false
        d_loss = (Lossfunc(discriminator(original_imgs), True_result) + Lossfunc(discriminator(generated_imgs.detach()), False_result)) / 2       # detach(): Prevent influence on gradient calculation of generator 
        d_loss.backward()      # compute derivatives
        dis_gd.step()          # do adam gradient descent
 
        print("epoch %d - batch %d " % (epoch, i))
 
        # save image every 400 batches in epoch
        if i % P_save_img_interval == 0:
            save_image(generated_imgs.data[:30], "gan_images/%d-%d.png" % (epoch, i), nrow=10)