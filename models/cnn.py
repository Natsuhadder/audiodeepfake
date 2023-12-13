import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels= 1, out_channels= 1,stride=1, kernel_size=3, padding=1,dilation=1)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(in_channels= 1, out_channels= 1,stride=1, kernel_size=3, padding=1,dilation=1)
        self.bn1 = nn.BatchNorm1d(1)
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(in_features= 128,out_features=1)

    def forward(self, x):  
        x = self.relu1(self.conv1(x)) # 32 x 256 x 128
        x = self.drop(x)
        x = self.relu2(self.bn1(self.conv2(x))) # 32 x 256 x 128
        x = self.linear(x) # 32 x 1
        
        x = nn.Flatten()(x)
        return x # 32 x 2