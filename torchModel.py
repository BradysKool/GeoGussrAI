#Note: This script is no longer used as I am now using a pre-trained model not my own

import torch
import torch.nn as nn
import torch.nn.functional as F

class Countryguessr(nn.Module):
    def __init__(self, numCountrys):
        super(Countryguessr,self).__init__()
        
        #Original image is 2560x1440 with 3 channels(RGB)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 32, kernel_size=3,stride=1) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dense1 = nn.Linear(512*43*78,128)#apply H_out and W_out formula 5 times to get final dimentions and multipy it by total out channels
        self.dense2 = nn.Linear(128,numCountrys)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = self.pool(x)

        x = torch.flatten(x,1) # get all to 1 dimention so can use standered nereal network
        x = F.relu(self.dense1(x))
        x = self.dense2(x) 

        return x