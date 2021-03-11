import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    """ An implementation of the Inception blocks desired in the GoogLeNet Paper """
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
