import torch
import numpy as np
import model.wgan as wgan
import torch.utils.data as data
from torchvision import datasets,transforms
from torch import optim
import visdom as vs
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
"""
load data from where you want
"""
def load_data(root):
    trans = transforms.Compose([
        transforms.Scale(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
    dataset = datasets.ImageFolder(root,transform=trans)
    dataloader = data.DataLoader(dataset,batch_size=5,shuffle=True)
    return dataloader
"""
generate random noise, can be thought as "ideas"
"""
def gen_noise():
    noise = torch.randn(batch_size,128)
    noise.to(device)
    return noise
"""
hyperparameters
"""
gene = wgan.Generate(128,3)
disc = wgan.Discrim(3)
epoch = 1000
LR = 1e-4
gene.to(device)
disc.to(device)
optimizer_g = optim.RMSprop(gene.parameters(),lr=LR)
optimizer_d = optim.RMSprop(disc.parameters(),lr=LR)
root = "/home/hjz/Pictures"
D_iters = 1
G_iters = 1
batch_size = 5

def train():
    vis = vs.Visdom(env=u'wgan')
    dataload = load_data(root)
    dataiter = iter(dataload)
    Loss_G = []
    Loss_D = []
    index = []
    for i in range(epoch):
        for p in gene.parameters():
            p.requires_grad = False
        for j in range(D_iters):
            noise = gen_noise()
            fakeimg = gene(noise)
            disc_fake = disc(fakeimg)
            img = dataiter.next()[0]
            disc_real = disc(img)
            # criter = wganloss(disc.parameters())
            # loss = criter(disc_fake,disc_real)
            loss_d = disc_real - disc_fake
            loss_d = loss_d.view(-1)
            loss_d = loss_d.sum()
            loss_d.backward()
            optimizer_d.step()
            if i%1000 == 999:
                Loss_D.append(loss_d.data)
        for p in disc.parameters():
            p.requires_grad = False
        for p in gene.parameters():
            p.requires_grad = True

        for j in range(G_iters):
            noise = gen_noise()
            fakeimg = gene(noise)
            loss_g = disc(fakeimg)
            loss_g = loss_g.view(-1)
            loss_g = loss_g.sum()
            loss_g.backward()
            optimizer_g.step()
            if i%1000 == 999:
                Loss_G.append(loss_g.data)
        for p in disc.parameters():
            p.requires_grad = True
    vis.line(X=index, Y=np.array(Loss_G), win='wgan', update='append')
    vis.line(X=index, Y=np.array(Loss_D), win='wgan', update='append')
train()
        


            
    