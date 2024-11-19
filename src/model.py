import torch
import torch.nn as nn
import torch.nn.functional as F

class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 8x28x28
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 16x14x14
        self.fc1 = nn.Linear(16 * 7 * 7, 10)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 8x14x14
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 16x7x7
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1) 