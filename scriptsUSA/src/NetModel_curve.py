#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
 @Author: HUJING
 @Time:5/14/18 11:18 AM 2018
 @Email: jhsa26@mail.ustc.edu.cn
 @Site:jhsa26.github.io
 """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# conv
# W_{n+1} = (W_{n} + Padding*2 - K_{w})/S + 1
# H_{n+1} = (H_{n} + Padding*2 - K_{h})/S + 1
# pooling
# W{n+1} = (W{n} - K{w})/S+1
# H{n+1} = (H{n} - K{h})/S
class Net(nn.Module):
    def __init__(self,image_width,
                 image_height,
                 image_outwidth,
                 image_outheight,
                 inchannel,
                 outchannel=4):
        super(Net,self).__init__()
        self.width = image_outwidth
        self.height = image_outheight
        self.conv1 = nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=3,stride=1,padding=1)  # 2 to 4
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(outchannel)

        self.conv2 = nn.Conv2d(in_channels=outchannel,out_channels=outchannel*2,kernel_size=3,stride=1,padding=1) #4 to 8
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(outchannel*2)
	#outchannel = outchannel*2
        self.conv3 = nn.Conv2d(in_channels=outchannel*2,out_channels=outchannel*4,kernel_size=3,stride=1,padding=1) # 8 to 16
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(outchannel*4)
	#outchannel = outchannel*2
        self.conv4 = nn.Conv2d(in_channels=outchannel*4,out_channels=outchannel*4,kernel_size=3,stride=1,padding=1) # 16 to 16 
        self.pool4 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(outchannel*4)
	#outchannel = outchannel*2
        self.fc1 = nn.Linear(in_features=outchannel*4*image_height*image_width, out_features=image_outheight*image_outwidth)
#        self.dropout = nn.Dropout2d(p=0.2)
    def forward(self, input):
        x = F.leaky_relu((self.conv1(input)))   
        x = F.leaky_relu((self.conv2(x)))      
        x = F.leaky_relu((self.conv3(x)))     
        x = F.leaky_relu((self.conv4(x)))    
        x = x.view(x.size(0),-1)
        x = (self.fc1(x))
        x = x.view(x.size(0), self.height*self.width)
        return x
if __name__ == '__main__':

    pass
