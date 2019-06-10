# -*- coding: utf-8 -*-
"""
main pgd enresnet
"""
import argparse
import os
import shutil
import time

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn as nn
import math

from resnet_cifar import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--model_name', default='en_wideresnet34_10_cifar10', type=str, help='name of the model')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--num-ensembles', '--ne', default=1, type=int, metavar='N')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--noise-coef', '--nc', default=0.1, type=float, metavar='W', help='forward noise (default: 0.1)')
parser.add_argument('--noise-coef-eval', '--nce', default=0.0, type=float, metavar='W', help='forward noise (default: 0.)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT',
                    help='10 for cifar10,100 for cifar100 (default: 10)')


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, noise_coef=None): # BW
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.noise_coef = noise_coef

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        
        if self.noise_coef is not None: # Test Variable and rand
            #return out + self.noise_coef * torch.std(out) + Variable(torch.randn(out.shape).cuda())
            return out + self.noise_coef * torch.std(out) * torch.randn_like(out)
        else:
            return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, noise_coef=None):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, noise_coef)
        self.noise_coef = noise_coef
    
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, noise_coef):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, noise_coef=noise_coef))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, noise_coef=None):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, noise_coef=noise_coef)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, noise_coef=noise_coef)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, noise_coef=noise_coef)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, noise_coef=noise_coef)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class AttackPGD(nn.Module):
    """
    PGD Adversarial training    
    """
    def __init__(self, basic_net, config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Only xent supported for now.'
    
    def forward(self, inputs, targets):
        x = inputs
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps): # iFGSM attack
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_net(x)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return self.basic_net(x), x


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available
    global best_acc
    best_acc = 0
    start_epoch = 0
    args = parser.parse_args()
    best_count = 0
    
    #--------------------------------------------------------------------------
    # Load Cifar data
    #--------------------------------------------------------------------------
    print('==> Preparing data...')
    root = './data'
    download = True
    
    #normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    
    
    train_set = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,
        ]))
    
    #train_set_tmp = train_set[:45000]
    #val_set = train_set[45000:]
    #train_set = train_set_tmp
    '''
    train_set_tmp = []; val_set = []
    for i in range(45000):
        train_set_tmp.append(train_set[i])
    for i in range(45000, 50000):
        val_set.append(train_set[i])
    train_set = train_set_tmp
    '''
    
    test_set = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            #normalize,
        ]))
    
    
    kwargs = {'num_workers':1, 'pin_memory':True}
    batchsize_test = len(test_set)/100#50 #100
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    batchsize_train = 128
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batchsize_train,
                                               shuffle=True, **kwargs
                                              )
    '''
    batchsize_val = 100
    print('Batch size of the validation set: ', batchsize_val)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                               batch_size=batchsize_val,
                                               shuffle=False, **kwargs
                                              )
    '''
    
    basic_net = WideResNet(noise_coef=args.noise_coef).cuda()
    
    # From https://github.com/MadryLab/cifar10_challenge/blob/master/config.json
    config = {
        'epsilon': 0.031, #8.0 / 255, # Test 1.0-8.0
        'num_steps': 10,
        'step_size': 0.007, #6.0 / 255, # 7.0
        'random_start': True,
        'loss_func': 'xent',
    }
    
    net = AttackPGD(basic_net, config).cuda()
    criterion = nn.CrossEntropyLoss()
    
    nepoch = 80
    for epoch in xrange(nepoch):
        print('Epoch ID', epoch)
        '''
        if epoch < 60:
            lr = 0.1
        elif epoch < 75:
            lr = 0.1/10
        elif epoch < 85:
            lr = 0.1/10/10
        else:
            lr = 0.1/10/10/10
        '''
        if epoch < 75:
            lr = 0.1
        elif epoch < 77:
            lr = 0.1/10
        elif epoch < 79:
            lr = 0.1/10/10
        else:
            lr = 0.1/10/10/10
        
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2, nesterov=True)
        
        #----------------------------------------------------------------------
        # Training
        #----------------------------------------------------------------------
        correct = 0; total = 0; train_loss = 0
        net.train()
        for batch_idx, (x, target) in enumerate(train_loader):
          if batch_idx < 352:
            optimizer.zero_grad()
            x, target = Variable(x.cuda()), Variable(target.cuda())
            
            score, pert_x = net(x, target)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data[0]
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
        #----------------------------------------------------------------------
        # Validation
        #----------------------------------------------------------------------
        val_loss = 0; correct = 0; total = 0
        net.eval()
        for batch_idx, (x, target) in enumerate(train_loader):
          if batch_idx >= 352:
            x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
            score, pert_x = net(x, target)
            
            loss = criterion(score, target)
            val_loss += loss.data[0]
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        #----------------------------------------------------------------------
        # Save the checkpoint
        #----------------------------------------------------------------------
        acc = 100.*correct/total
        #if acc > best_acc:
        if correct > best_count:
            print('Saving model...')
            state = {
                'net': basic_net, #net,
                'acc': acc,
                'epoch': epoch,
            }
            
            torch.save(state, './ckpt_PGD_ensemble_WideResNet.t7')
            #best_acc = acc
            #best_count = correct
        
        #----------------------------------------------------------------------
        # Testing
        #----------------------------------------------------------------------
        if correct > best_count:
          best_count = correct
          test_loss = 0; correct = 0; total = 0
          net.eval()
          for batch_idx, (x, target) in enumerate(test_loader):
            x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
            score, pert_x = net(x, target)
            
            loss = criterion(score, target)
            test_loss += loss.data[0]
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('The best acc: ', best_count)
