# -*- coding: utf-8 -*-
"""
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PreActBasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, noise_coef=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.noise_coef = noise_coef
    
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        
        if self.downsample is not None:
            residual = self.downsample(out)
        
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += residual
        
        if self.noise_coef is not None: # Test Variable and rand
            #return out + self.noise_coef * torch.std(out) + Variable(torch.randn(out.shape).cuda())
            return out + self.noise_coef * torch.std(out) * torch.randn_like(out)
        else:
            return out


class PreActBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, noise_coef=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.noise_coef = noise_coef
    
    def forward(self, x):
        residual = x
        
        out = self.bn1(x)
        out = self.relu(out)
        
        if self.downsample is not None:
            residual = self.downsample(out)
        
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        out += residual
        if self.noise_coef is not None:
            #return out + self.noise_coef * torch.std(out) * Variable(torch.randn(out.shape).cuda())
            return out + self.noise_coef * torch.std(out) * torch.randn_like(out)
        else:
            return out


class PreAct_ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10, noise_coef=None):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0], noise_coef=noise_coef)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, noise_coef=noise_coef)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, noise_coef=noise_coef)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)
        
        #self.loss = nn.CrossEntropyLoss()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride=1, noise_coef=None):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, noise_coef=noise_coef))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, noise_coef=noise_coef))
        return nn.Sequential(*layers)
    
    #def forward(self, x, target):
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        #loss = self.loss(x, target)
        
        #return x, loss
        return x


class Ensemble_PreAct_ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_ensembles=3, noise_coef=0.0):
        super(Ensemble_PreAct_ResNet_Cifar, self).__init__()
        self.num_ensembles = num_ensembles
        # for emsemble resnet we should use Noisy Blocks.
        self.ensemble = nn.ModuleList([PreAct_ResNet_Cifar(block, layers, num_classes=num_classes, noise_coef=noise_coef) for i in range(num_ensembles)])
        # self.ensemble = nn.ModuleList([ResNet_Cifar(block, layers, num_classes=num_classes) for i in range(num_ensembles)])
    
    def forward(self, x):
    #def forward(self, x, target):
        ret = 0.0
        for net in self.ensemble:
            ret += net(x)
            #ret += net(x, target)
        ret /= self.num_ensembles
        
        return ret


def en_preactresnet20_cifar(**kwargs):
    model = Ensemble_PreAct_ResNet_Cifar(PreActBasicBlock, [3, 3, 3], **kwargs)
    return model

def en_preactresnet44_cifar(**kwargs):
    model = Ensemble_PreAct_ResNet_Cifar(PreActBasicBlock, [7, 7, 7], **kwargs)
    return model

def en_preactresnet32_cifar(**kwargs):
    model = Ensemble_PreAct_ResNet_Cifar(PreActBasicBlock, [5, 5, 5], **kwargs)
    return model

def en_preactresnet110_cifar(**kwargs):
    model = Ensemble_PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


if __name__ == '__main__':
    net = en_preactresnet20_cifar()
    y = net(torch.autograd.Variable(torch.randn(1, 3, 32, 32)))
    print(net)
    print(y.size())
