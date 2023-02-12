#!/usr/bin/env python
# coding=utf-8
'''
Author: zm, wjm
Date: 2020-11-11 20:37:09
LastEditTime: 2020-12-09 23:12:50
Description: Reverse Filter for pan-sharpening
batch_size = 64, MSE, Adam, 0.0001, patch_size = 64, 2000 epoch, decay 1000, x0.1
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F
from .arfnet_layers import *

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()

        kernel_mode='TG'#'FG', 
        num_gaussian_kernels=34 #42
        gaussian_kernel_size=17#21

        self.Ffunction=nn.ModuleList()
        self.Gfunction=nn.ModuleList()
        self.iter=4
        module_num=1
        for j in range( self.iter+1):
            Ffunction=[]
            Gfunction=[]
            for i in range(module_num):
                Ffunction.append(GaussianBlurLayer(num_gaussian_kernels, gaussian_kernel_size, kernel_mode,channels=4))
                Ffunction.append(ConvBlock(140, 88, 1, 1, 0, activation='prelu', norm=None, bias = False))
                Ffunction.append(ConvBlock(88, 44, 1, 1, 0, activation='prelu', norm=None, bias = False))
                Ffunction.append(nn.Conv2d(44,4,1,1,0,bias=False)) 
                if j != self.iter:
                    Gfunction.append(GaussianBlurLayer(num_gaussian_kernels, gaussian_kernel_size, kernel_mode,channels=1))
                    Gfunction.append(ConvBlock(35, 22, 1, 1, 0, activation='prelu', norm=None, bias = False))
                    Gfunction.append(ConvBlock(22, 11, 1, 1, 0, activation='prelu', norm=None, bias = False))
                    Gfunction.append(nn.Conv2d(11,1,1,1,0,bias=False))
            
            self.Ffunction.append(nn.Sequential(*Ffunction))
            if j != self.iter:
                self.Gfunction.append(nn.Sequential(*Gfunction))
        
        self.refine=nn.Conv2d(1,4,1,1,0,bias=False)


    def forward(self, l_ms, b_ms, x_pan):
        Hk=b_ms
        L=b_ms
        P=x_pan
        _,C,_,_ = P.shape
        I=torch.mean(Hk,axis=1,keepdim=True)
        
        for i in range(self.iter):
            # print(self.Ffunction(Hk).shape)#[1, 88, 128, 128]--->[1,4,128,128]
            Hk=Hk+(L-self.Ffunction[i](Hk))
            # print(Hk.shape)
            # I=np.mean(Hk, axis=1, keepdims=True)
            I=torch.mean(Hk,axis=1,keepdim=True)
            # print(I.shape) #(1,1,128,128)
            # print(self.Gfunction(I).shape)
            I=I+(P-self.Gfunction[i](I))
            # print(I.shape) #(1,1,128,128)
            # print(torch.mean(P))
            Pt = (P - torch.mean(P))*torch.std(I)/torch.std(P)+torch.mean(I) #np.std(I, ddof=1)
            # print(Pt.shape)#[1, 1, 128, 128]
            Hk = Hk + (Pt-I)#torch.tile(Pt-I, (1, 1, C)) #np.tile
            # Hk = Hk + self.refine(Pt-I)
        Hk=Hk+(L-self.Ffunction[4](Hk))
        return Hk,I
