import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Generator(nn.Module):
    def __init__(self, input_cns=1, output_cns=1, n_blocks=9):
        self.n_blocks = n_blocks
        self.input_cns = input_cns
        self.output_cns = output_cns

        self.conv1 = nn.Conv2d(in_channels=self.input_cns, out_channels=64, kernel_size=7, stride=1, padding=0)
        self.conv1_norm = nn.InstanceNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, 2, 0)
        self.conv2_norm = nn.InstanceNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(256)

        self.resnet_blocks = []
        for i in range(self.n_blocks):
            self.resnet_blocks.append(ResnetBlock(256, 3, 1, 1))
            self.resnet_blocks[i].weight_init(0, 0.02)
        
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv1_norm = nn.InstanceNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv2_norm = nn.InstanceNorm2d(64)
        self.deconv3 = nn.Conv2d(64, self.output_cns, 7, 1, 0)

        self.Tanh = nn.Tanh()

    def forward(self, input):
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks(x)
        x = F.relu(self.deconv1_norm(self.deconv1(x)))
        x = F.relu(self.deconv2_norm(self.deconv2(x)))
        x = F.pad(x, (3, 3, 3, 3), 'reflect')
        x = self.deconv3(x)
        image = self.tanh(x)
        return image



def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class ResnetBlock(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(ResnetBlock, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)

        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        x = self.conv2_norm(self.conv2(x))
        return input + x


class Discriminator(nn.Module):
    

