import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.Inception import *

class Inception_v3(nn.Module):
    def __init__(self, n_classes):
        """ A Pytorch implementation of the inception v3 architecture decibed in the paper
            "Rethinking the Inception Architecture for Computer Vision"  by Szegedy et al.""" 
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64,80,kernel_size=3,stride=1)
        self.conv5 = nn.Conv2d(80,192,kernel_size=3,stride=2)
        self.conv6 = nn.Conv2d(192,288,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.inca1 = FactorInceptionA(288,192,(235,192),(235,192),192)
        self.inca2 = FactorInceptionA(768,192,(192,192),(192,192),192)
        self.inca3 = FactorInceptionA(768,192,(192,192),(192,192),192)
        self.incb1 = FactorInceptionB(768,320,(480,320),(480,320),320,7)
        self.incb2 = FactorInceptionB(1280,320,(320,320),(320,320),320,7)
        self.incb3 = FactorInceptionB(1280,320,(320,320),(320,320),320,7)
        self.incb4 = FactorInceptionB(1280,320,(320,320),(320,320),320,7)
        self.incb5 = FactorInceptionB(1280,320,(320,320),(320,320),320,7)
        self.incc1 = FactorInceptionC(1280,512,(768,512),(768,512),512)
        self.incc2 = FactorInceptionC(2048,512,(512,512),(512,512),512)
        self.pool2 = nn.MaxPool2d(kernel_size=8)
        self.fc1   = nn.Linear(2048,n_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.inca1(x)
        x = self.inca2(x)
        x = self.inca3(x)
        x = self.pool1(x)
        x = self.incb1(x)
        x = self.incb2(x)
        x = self.incb3(x)
        x = self.incb4(x)
        x = self.incb5(x)
        x = self.pool1(x)
        x = self.incc1(x)
        x = self.incc2(x)
        x = self.pool2(x)
        x = x.view(-1,2048) 
        x = self.fc1(x)
        return x
