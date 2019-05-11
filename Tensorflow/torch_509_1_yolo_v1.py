import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchsummary.torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

import visdom


class YOLOV1(nn.Module):
    def __init__(self,params):
        self.dropout_prop= params["dropout"]
        self.num_classes=params["num_class"]

        super(YOLOV1,self).__init__()
        #layer1
        self.layer1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))

        #layer2
        self.layer2=nn.Sequential(
            nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(192,momentum=0.01),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        #layer3
        self.layer3=nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.LeakyReLU())
        #layer4
        self.layer4=nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256,momentum=0.01),
            nn.LeakyReLU())
        #layer5
        self.



