import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, floor

class GoogLeNetInception(nn.Module):
    """ An implementation of the Inception block described in the GoogLeNet Paper """
    def __init__(self, dim_in, dim_1, dims_3, dims_5, dim_pool):
        super().__init__()
        # 1x1 Convolution Path
        self.redu1 = nn.Conv2d(dim_in, dim_1, kernel_size=1, stride=1) 
        # 3x3 Convolution Path
        self.redu2 = nn.Conv2d(dim_in, dims_3[0], kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(dims_3[0], dims_3[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        # 5x5 Convolution Path
        self.redu3 = nn.Conv2d(dim_in, dims_5[0], kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(dims_5[0], dims_5[1], kernel_size=5, stride=1, padding=2, padding_mode='reflect')
        # Max-Pool Pat
        self.pool  = nn.MaxPool2d(3,1, padding=1)
        self.redu4 = nn.Conv2d(dim_in, dim_pool, kernel_size=1,stride=1)

    def forward(self,x):
        # 1x1 Convolution Path
        x1 = F.relu(self.redu1(x))
        
        # 3x3 Convolution Path
        x2 = F.relu(self.redu2(x))
        x2 = F.relu(self.conv1(x2))
        # 5x5 Convolution Path
        x3 = F.relu(self.redu3(x))
        x3 = F.relu(self.conv2(x3))
        
        # Max-Pool Path
        x4 = self.pool(x)
        x4 = F.relu(self.redu4(x4))
        
        #  Concatenate the resulting tensors
        x = torch.cat((x1,x2,x3,x4),1) 
        return x

class FactorInceptionA(nn.Module):
    """ An implementation of the Factorized Inception block described in Figure 5 
        of the paper "Rethinking the Inception Architecture for Computer Vision" """
    def __init__(self, dim_in, dim_1, dims_3, dims_5, dim_pool):
        super().__init__()
        # 1x1 Convolution Path
        self.redu1 = nn.Conv2d(dim_in, dim_1, kernel_size=1, stride=1) 
        # 3x3 Convolution Path
        self.redu2 = nn.Conv2d(dim_in, dims_3[0], kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(dims_3[0], dims_3[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        # 5x5 Convolution Path
        self.redu3 = nn.Conv2d(dim_in, dims_5[0], kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(dims_5[0], dims_5[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(dims_5[1], dims_5[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        # Max-Pool Pat
        self.pool  = nn.MaxPool2d(3,1, padding=1)
        self.redu4 = nn.Conv2d(dim_in, dim_pool, kernel_size=1,stride=1)

    def forward(self,x):
        # 1x1 Convolution Path
        x1 = F.relu(self.redu1(x))
        
        # 3x3 Convolution Path
        x2 = F.relu(self.redu2(x))
        x2 = F.relu(self.conv1(x2))
        # 5x5 Convolution Path
        x3 = F.relu(self.redu3(x))
        x3 = F.relu(self.conv2(x3))
        x3 = F.relu(self.conv3(x3))
        
        # Max-Pool Path
        x4 = self.pool(x)
        x4 = F.relu(self.redu4(x4))
        
        #  Concatenate the resulting tensors
        x = torch.cat((x1,x2,x3,x4),1) 
        return x

class FactorInceptionB(nn.Module):
    """ An implementation of the Factorized Inception blocks described in Figure 6 
        of the paper "Rethinking the Inception Architecture for Computer Vision" 
        padding configuration currently somewhat... Problematic?"""
    def __init__(self, dim_in, dim_1, dims_3, dims_5, dim_pool, n):
        super().__init__()
        # 1x1 Convolution Path
        self.redu1 = nn.Conv2d(dim_in, dim_1, kernel_size=1, stride=1) 
        
        # nxn Convolution Path
        self.redu2 = nn.Conv2d(dim_in, dims_3[0], kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(dims_3[0], dims_3[1], kernel_size=(1,n), stride=1, padding=(0,3), padding_mode='reflect')
        self.conv2 = nn.Conv2d(dims_3[1], dims_3[1], kernel_size=(n,1), stride=1, padding=(3,0), padding_mode='reflect')
        
        # ?x? Convolution Path
        self.redu3 = nn.Conv2d(dim_in, dims_5[0], kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(dims_3[0], dims_3[1], kernel_size=(1,n), stride=1, padding=(0,3), padding_mode='reflect')
        self.conv4 = nn.Conv2d(dims_3[1], dims_3[1], kernel_size=(n,1), stride=1, padding=(3,0), padding_mode='reflect')
        self.conv5 = nn.Conv2d(dims_3[1], dims_3[1], kernel_size=(1,n), stride=1, padding=(0,3), padding_mode='reflect')
        self.conv6 = nn.Conv2d(dims_3[1], dims_3[1], kernel_size=(n,1), stride=1, padding=(3,0), padding_mode='reflect')
        
        # Max-Pool Pat
        self.pool  = nn.MaxPool2d(3,1, padding=1)
        self.redu4 = nn.Conv2d(dim_in, dim_pool, kernel_size=1,stride=1)

    def forward(self,x):
        # 1x1 Convolution Path
        x1 = F.relu(self.redu1(x))
        
        # nxn Convolution Path
        x2 = F.relu(self.redu2(x))
        x2 = F.relu(self.conv1(x2))
        x2 = F.relu(self.conv2(x2))

        # ?x? Convolution Path
        x3 = F.relu(self.redu3(x))
        x3 = F.relu(self.conv3(x3))
        x3 = F.relu(self.conv4(x3))
        x3 = F.relu(self.conv5(x3))
        x3 = F.relu(self.conv6(x3))

        # Max-Pool Path
        x4 = self.pool(x)
        x4 = F.relu(self.redu4(x4))
        
        #  Concatenate the resulting tensors
        x = torch.cat((x1,x2,x3,x4),1) 
        return x

class FactorInceptionC(nn.Module):
    """ An implementation of the Factorized Inception blocks described in Figure 7 
        of the paper "Rethinking the Inception Architecture for Computer Vision" """
    def __init__(self, dim_in, dim_1, dims_3, dims_5, dim_pool):
        super().__init__()
        # 1x1 Convolution Path
        self.redu1 = nn.Conv2d(dim_in, dim_1, kernel_size=1, stride=1) 
        # 3x3 Convolution Path 
        self.redu3 = nn.Conv2d(dim_in, dims_3[0], kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(dims_3[0], dims_3[1]//2, kernel_size=(1,3), stride=1, padding=(0,1), padding_mode='reflect')
        self.conv2 = nn.Conv2d(dims_3[0], dims_3[1]//2, kernel_size=(3,1), stride=1, padding=(1,0), padding_mode='reflect')
        # 5x5 Convolution Path
        
        self.redu2 = nn.Conv2d(dim_in, dims_5[0], kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(dims_5[0], dims_5[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv2d(dims_5[1], dims_5[1]//2, kernel_size=(1,3), stride=1, padding=(0,1), padding_mode='reflect')
        self.conv5 = nn.Conv2d(dims_5[1], dims_5[1]//2, kernel_size=(3,1), stride=1, padding=(1,0), padding_mode='reflect')
        # Max-Pool Pat
        self.pool  = nn.MaxPool2d(3,1, padding=1)
        self.redu4 = nn.Conv2d(dim_in, dim_pool, kernel_size=1,stride=1)

    def forward(self,x):
        # 1x1 Convolution Path
        x1 = F.relu(self.redu1(x))
        
        # 3x3 Convolution Path
        x2 = F.relu(self.redu2(x))
        x2a = F.relu(self.conv1(x2))
        x2b = F.relu(self.conv2(x2))

        # 5x5 Convolution Path
        x3 = F.relu(self.redu3(x))
        x3 = F.relu(self.conv3(x3))
        x3a = F.relu(self.conv4(x3))
        x3b = F.relu(self.conv5(x3))
        
        # Max-Pool Path
        x4 = self.pool(x)
        x4 = F.relu(self.redu4(x4))
        
        #  Concatenate the resulting tensors
        x = torch.cat((x1,x2a,x2b,x3a,x3b,x4),1) 
        return x

class FactorInception4(nn.Module):
    """ An implementation of the Factorized Inception blocks described in Figure 5 
        of the paper "Rethinking the Inception Architecture for Computer Vision" """
    def __init__(self, dim_in, dim_1, dims_3, dims_5, dim_pool):
        super().__init__()
        # 1x1 Convolution Path
        self.redu1 = nn.Conv2d(dim_in, dim_1, kernel_size=1, stride=1) 
        # 3x3 Convolution Path
        self.redu2 = nn.Conv2d(dim_in, dims_3[0], kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(dims_3[0], dims_3[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        # 5x5 Convolution Path
        self.redu3 = nn.Conv2d(dim_in, dims_5[0], kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(dims_5[0], dims_5[1], kernel_size=5, stride=1, padding=2, padding_mode='reflect')
        # Max-Pool Pat
        self.pool  = nn.MaxPool2d(3,1, padding=1)
        self.redu4 = nn.Conv2d(dim_in, dim_pool, kernel_size=1,stride=1)

    def forward(self,x):
        # 1x1 Convolution Path
        x1 = F.relu(self.redu1(x))
        
        # 3x3 Convolution Path
        x2 = F.relu(self.redu2(x))
        x2 = F.relu(self.conv1(x2))
        # 5x5 Convolution Path
        x3 = F.relu(self.redu3(x))
        x3 = F.relu(self.conv2(x3))
        
        # Max-Pool Path
        x4 = self.pool(x)
        x4 = F.relu(self.redu4(x4))
        
        #  Concatenate the resulting tensors
        x = torch.cat((x1,x2,x3,x4),1) 
        return x
