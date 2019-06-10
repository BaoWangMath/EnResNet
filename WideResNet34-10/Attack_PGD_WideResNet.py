# -*- coding: utf-8 -*-
"""
CW, FGSM, and IFGSM Attack CNN
"""
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import copy
import math
import numpy as np
import os
import argparse

#from utils import *

import numpy.matlib
import matplotlib.pyplot as plt
import pickle
import cPickle
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Fool ResNet_PGD ')
ap = parser.add_argument
ap('-method', help='Attack Method', type=str, default="ifgsm") # fgsm, ifgsm, cwl2
#ap('-epsilon', help='Attack Strength', type=float, default=0.007)
ap('-epsilon', help='Attack Strength', type=float, default=0.003) # May 2
#ap('-epsilon', help='Attack Strength', type=float, default=0.031) # May 2
ap('--num-ensembles', '--ne', default=1, type=int, metavar='N')
ap('--noise-coef', '--nc', default=0.0, type=float, metavar='W', help='forward noise (default: 0.0)')
ap('--noise-coef-eval', '--nce', default=0.0, type=float, metavar='W', help='forward noise (default: 0.)')
opt = vars(parser.parse_args())


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

if __name__ == '__main__':
    """
    Load the trained DNN, and attack the DNN, finally save the adversarial images
    """
    # Load the model
    print '==> Resuming from checkpoint..'
    checkpoint = torch.load('ckpt_PGD_ensemble_WideResNet.t7')
    net = checkpoint['net']
    epsilon = opt['epsilon']
    attack_type = opt['method']
    
    # Load the original test data
    print '==> Load the clean image'
    root = './data'
    download = False
    
    test_set = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            #normalize,
        ]))
    
    kwargs = {'num_workers':1, 'pin_memory':True}
    batchsize_test = 50
    if attack_type == 'cw':
        batchsize_test = 1
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    criterion = nn.CrossEntropyLoss()
    #--------------------------------------------------------------------------
    # Testing
    # images: the original images
    # labels: labels of the original images
    # images_adv: adversarial image
    # labels_pred: the predicted labels of the adversarial images
    # noise: the added noise
    #--------------------------------------------------------------------------
    images, labels, images_adv, labels_pred, noise = [], [], [], [], []
    total_fooled = 0; total_correct_classified = 0
    
    if attack_type == 'fgsm':
        for batch_idx, (x1, y1_true) in enumerate(test_loader):
          #if batch_idx < 2:
            x_Test = x1.numpy()
            #print x_Test.min(), x_Test.max()
            #x_Test = ((x_Test - x_Test.min())/(x_Test.max() - x_Test.min()) - 0.5)*2
            #x_Test = (x_Test - x_Test.min() )/(x_Test.max() - x_Test.min())
            y_Test = y1_true.numpy()
            
            #x = Variable(torch.cuda.FloatTensor(x_Test.reshape(1, 1, 28, 28)), requires_grad=True)
            x = Variable(torch.cuda.FloatTensor(x_Test.reshape(batchsize_test, 3, 32, 32)), requires_grad=True)
            y = Variable(torch.cuda.LongTensor(y_Test), requires_grad=False)
            
            # Classification before perturbation
            pred_tmp = net(x)
            y_pred = np.argmax(pred_tmp.cpu().data.numpy())
            loss = criterion(pred_tmp, y)
            # Attack
            net.zero_grad()
            if x.grad is not None:
                x.grad.data.fill_(0)
            loss.backward()
            
            x_val_min = 0.0
            x_val_max = 1.0
            x.grad.sign_()
            
            x_adversarial = x + epsilon*x.grad
            x_adversarial = torch.clamp(x_adversarial, x_val_min, x_val_max)
            x_adversarial = x_adversarial.data
            
            # Classify the perturbed data
            x_adversarial_tmp = Variable(x_adversarial)
            pred_tmp = net(x_adversarial_tmp)
            y_pred_adversarial = np.argmax(pred_tmp.cpu().data.numpy(), axis=1)
            
            for i in range(len(x_Test)):
                #print y_pred_adversarial
                if y_Test[i] == y_pred_adversarial[i]:
                    #if y_Test == y_pred_adversarial:
                    total_correct_classified += 1
            
            for i in range(len(x_Test)):
                # Save the perturbed data
                images.append(x_Test[i, :, :, :]) # Original image
                images_adv.append(x_adversarial.cpu().numpy()[i, :, :, :]) # Perturbed image
                noise.append(x_adversarial.cpu().numpy()[i, :, :, :]-x_Test[i, :, :, :]) # Noise
                labels.append(y_Test[i])
                labels_pred.append(y_pred_adversarial[i])
    
    elif attack_type == 'ifgsm':
        for batch_idx, (x1, y1_true) in enumerate(test_loader):
          #if batch_idx < 100:
            x_Test = x1.numpy()
            y_Test = y1_true.numpy()
            
            x = Variable(torch.cuda.FloatTensor(x_Test.reshape(batchsize_test, 3, 32, 32)), requires_grad=True)
            y = Variable(torch.cuda.LongTensor(y_Test), requires_grad=False)
            
            # Classification before perturbation
            pred_tmp = net(x)
            y_pred = np.argmax(pred_tmp.cpu().data.numpy())
            loss = criterion(pred_tmp, y)
            # Attack
            alpha = epsilon
            #iteration = 10
            iteration = 20#40 # May 2
            x_val_min = 0.; x_val_max = 1.
            epsilon1 = 0.031
            
            # Helper function
            def where(cond, x, y):
                """
                code from :
                https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
                """
                cond = cond.float()
                return (cond*x) + ((1-cond)*y)

            # Random perturbation
            #x = x + torch.zeros_like(x).uniform_(-epsilon1, epsilon1) # May 2
            x_adv = Variable(x.data, requires_grad=True)

            for i in range(iteration):
                #h_adv = net(x_adv)
                h_adv = (net(x_adv) + net(x_adv) + net(x_adv) + net(x_adv) + net(x_adv) + net(x_adv) + net(x_adv) + net(x_adv) + net(x_adv) + net(x_adv))/10.
                loss = criterion(h_adv, y)
                net.zero_grad()
                if x_adv.grad is not None:
                    x_adv.grad.data.fill_(0)
                loss.backward()
                
                x_adv.grad.sign_()
                x_adv = x_adv + alpha*x_adv.grad
                x_adv = where(x_adv > x+epsilon1, x+epsilon1, x_adv)
                x_adv = where(x_adv < x-epsilon1, x-epsilon1, x_adv)
                x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
                x_adv = Variable(x_adv.data, requires_grad=True)
                
            x_adversarial = x_adv.data
            
            x_adversarial_tmp = Variable(x_adversarial)
            pred_tmp = net(x_adversarial_tmp)
            loss = criterion(pred_tmp, y)
            y_pred_adversarial = np.argmax(pred_tmp.cpu().data.numpy(), axis=1)
            
            #if y_Test == y_pred_adversarial:
            #    total_correct_classified += 1
            for i in range(len(x_Test)):
                #print y_pred_adversarial
                if y_Test[i] == y_pred_adversarial[i]:
                    #if y_Test == y_pred_adversarial:
                    total_correct_classified += 1
            
            for i in range(len(x_Test)):
                # Save the perturbed data
                images.append(x_Test[i, :, :, :]) # Original image
                images_adv.append(x_adversarial.cpu().numpy()[i, :, :, :]) # Perturbed image
                noise.append(x_adversarial.cpu().numpy()[i, :, :, :]-x_Test[i, :, :, :]) # Noise
                labels.append(y_Test[i])
                labels_pred.append(y_pred_adversarial[i])
        
    elif attack_type == 'cw':
        for batch_idx, (x1, y1_true) in enumerate(test_loader):
          #if batch_idx < 4000:
            if batch_idx - int(int(batch_idx/50.)*50) == 0:
                print batch_idx
            x_Test = x1.numpy()
            y_Test = y1_true.numpy()
            
            x = Variable(torch.cuda.FloatTensor(x_Test.reshape(batchsize_test, 3, 32, 32)), requires_grad=True)
            y = Variable(torch.cuda.LongTensor(y_Test), requires_grad=False)
            
            # Classification before perturbation
            pred_tmp = net(x)
            loss = criterion(pred_tmp, y)
            y_pred = np.argmax(pred_tmp.cpu().data.numpy())
            
            # Attack
            cwl2_learning_rate = 0.0006#0.01
            max_iter = 50
            lambdaf = 10.0
            kappa = 0.0
            
            # The input image we will perturb 
            input = torch.FloatTensor(x_Test.reshape(batchsize_test, 3, 32, 32))
            input_var = Variable(input)
            
            # w is the variable we will optimize over. We will also save the best w and loss
            w = Variable(input, requires_grad=True) 
            best_w = input.clone()
            best_loss = float('inf')
            
            # Use the Adam optimizer for the minimization
            optimizer = optim.Adam([w], lr=cwl2_learning_rate)
            
            # Get the top2 predictions of the model. Get the argmaxes for the objective function
            probs = net(input_var.cuda())
            
            probs_data = probs.data.cpu()
            top1_idx = torch.max(probs_data, 1)[1]
            probs_data[0][top1_idx] = -1 # making the previous top1 the lowest so we get the top2
            top2_idx = torch.max(probs_data, 1)[1]
            
            # Set the argmax (but maybe argmax will just equal top2_idx always?)
            argmax = top1_idx[0]
            if argmax == y_pred:
                argmax = top2_idx[0]
                
            # The iteration
            for i in range(0, max_iter):
                if i > 0:
                    w.grad.data.fill_(0)
                
                # Zero grad (Only one line needed actually)
                net.zero_grad()
                optimizer.zero_grad()
                
                # Compute L2 Loss
                loss = torch.pow(w - input_var, 2).sum()
                
                # w variable
                w_data = w.data
                w_in = Variable(w_data, requires_grad=True)
                
                # Compute output
                output = net.forward(w_in.cuda()) #second argument is unneeded
                
                # Calculating the (hinge) loss
                loss += lambdaf * torch.clamp( output[0][y_pred] - output[0][argmax] + kappa, min=0).cpu()
                
                # Backprop the loss
                loss.backward()
                
                # Work on w (Don't think we need this)
                w.grad.data.add_(w_in.grad.data)
                
                # Optimizer step
                optimizer.step()
                
                # Save the best w and loss
                total_loss = loss.data.cpu()[0]
                
                if total_loss < best_loss:
                    best_loss = total_loss
                    
                    ##best_w = torch.clamp(best_w, 0., 1.) # BW Added Aug 26
                    
                    best_w = w.data.clone()
            
            # Set final adversarial image as the best-found w
            x_adversarial = best_w
            
            ##x_adversarial = torch.clamp(x_adversarial, 0., 1.) # BW Added Aug 26
            
            #--------------- Add to introduce the noise
            noise_tmp = x_adversarial.cpu().numpy() - x_Test
            x_adversarial = x_Test + epsilon * noise_tmp
            #---------------
            
            # Classify the perturbed data
            x_adversarial_tmp = Variable(torch.cuda.FloatTensor(x_adversarial), requires_grad=False) #Variable(x_adversarial).cuda()
            pred_tmp = net(x_adversarial_tmp)
            y_pred_adversarial = np.argmax(pred_tmp.cpu().data.numpy()) # axis=1
            
            if y_Test == y_pred_adversarial:
                total_correct_classified += 1
                        
            # Save the perturbed data
            images.append(x_Test) # Original image
            images_adv.append(x_adversarial) # Perturbed image
            noise.append(x_adversarial-x_Test) # Noise
            labels.append(y_Test)
            labels_pred.append(y_pred_adversarial)
    else:
        ValueError('Unsupported Attack')
    
    print('Number of correctly classified images: ', total_correct_classified)
    # Save data
    #with open("Adversarial" + attack_type + str(int(10*epsilon)) + ".pkl", "w") as f:
    #with open("Adversarial" + attack_type + str(int(100*epsilon)) + ".pkl", "w") as f:
    #    adv_data_dict = {"images":images_adv, "labels":labels}
    #    cPickle.dump(adv_data_dict, f)
    images = np.array(images).squeeze()
    images_adv = np.array(images_adv).squeeze()
    noise = np.array(noise).squeeze()
    labels = np.array(labels).squeeze()
    labels_pred = np.array(labels_pred).squeeze()
    print images.shape, images_adv.shape, noise.shape, labels.shape, labels_pred.shape
    
    with open("fooled_WideResNet_PGD_10iters_" + attack_type + str(int(1000*epsilon)) + ".pkl", "w") as f:
    #with open("fooled_WideResNet110_PGD_20iters_" + attack_type + str(int(1000*epsilon)) + ".pkl", "w") as f:
        adv_data_dict = {
            "images" : images,
            "images_adversarial" : images_adv,
            "y_trues" : labels,
            "noises" : noise,
            "y_preds_adversarial" : labels_pred
            }
        pickle.dump(adv_data_dict, f)
    
