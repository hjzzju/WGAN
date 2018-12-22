import torch
import torch.nn as nn
import numpy as np

class Downforward(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,H):
        super(Downforward,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.bn1 = nn.LayerNorm([in_channels,H,H])
        self.bn2 = nn.LayerNorm([in_channels,H,H])
        self.conv_1 = nn.Conv2d(in_channels,in_channels,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)
        self.conv_2 = nn.Conv2d(in_channels,out_channels,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)

    def forward(self,input):
        output = self.bn1(input)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return output


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

# '''
# 注意思路，(N,C,H,W) -> (N,H,W,4,C/4) -> (N,H,W,2,C/4) -> (N,H,W*2,C/4) -> (2,N,H,W*2,C/4) -> (N,H,2,W*2,C/4)
# -> (N,H*2,W*2,C/4) 当进行(H,2)->(H*2)时，必须二者相邻近，且(2,H)->(2*H)!=(H*2)<-(H,2)，所以要一步一步进行，想清楚维度变换，才能不出错
# '''
    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
        output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size,output_height,output_width,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output

class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,bias=True):
        super(UpSampleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,stride=1,padding=1, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class Upforward(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(Upforward,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_1 = UpSampleConv(in_channels,out_channels,kernel_size=self.kernel_size,bias=False)
        self.conv_2 = nn.Conv2d(out_channels,out_channels,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)    
    def forward(self,input):
        output = self.conv_2(self.relu2(self.bn2(self.conv_1(self.relu1(self.bn1(input))))))
        return output
   
class Discrim(nn.Module):
    def __init__(self,in_dim):
        super(Discrim,self).__init__()
        self.in_dim = in_dim
        self.downward = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(self.in_dim,128,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(512,1024,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024,1,kernel_size=4,stride=1,padding=0),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.downward(input)
        return output

class Generate(nn.Module):
    def __init__(self, in_vec,in_dim):
        super(Generate,self).__init__()
        self.in_vec = in_vec
        self.in_dim = in_dim
        self.upward = nn.Sequential(
            nn.ConvTranspose2d(self.in_vec,1024,4,stride=1,padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024,512,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,self.in_dim,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        N,L = np.array(input).shape
        input = torch.tensor(input)
        output = self.upward(input.view(N,L,1,1))
        return output