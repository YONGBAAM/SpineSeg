import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

################################
#
#       Refactoring Finished
#
################################
class CCB(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(CCB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= input_channel, out_channels=output_channel,
                          kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel,
                          kernel_size=3, padding = 1)
        self.bn1 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x=  F.relu(self.bn1(self.conv2(x)))
        return x

class CCU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(CCU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= input_channel, out_channels=output_channel,
                          kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel,
                          kernel_size=3, padding = 1)
        self.upconv1 = nn.Upsample(scale_factor=2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.upconv1(x)
        return x

class SegmentNet(nn.Module):
    def __init__(self):
        super(SegmentNet, self).__init__()
        self.ccb1 = CCB(1,16)

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.ccb2 = CCB(16,32)

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.ccb3 = CCB(32,64)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.ccb4 = CCB(64,64)

        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.ccb5 = CCB(64,64)
        self.ups = nn.Upsample(scale_factor=2)##나온거 테스트 보고 하기

        self.ccu1 = CCU(128,16)
        self.ccu2 = CCU(80,16)
        self.ccu3 = CCU(48,16)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding = 1)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        #Testing for size
        y1 = self.ccb1(x) # 508 252

        y2 = self.pool1(y1)
        y2 = self.ccb2(y2) #250 122

        y3 = self.pool2(y2)
        y3 = self.ccb3(y3) #121 57

        y4 = self.pool3(y3)
        y4 = self.ccb4(y4) #56 24

        y5 = self.pool4(y4)
        y5 = self.ccb5(y5) #chw 64 24 8

        y5 = self.ups(y5)
        y44 = torch.cat((y5,y4), dim = 1)
        y44 = self.ccu1(y44)

        y33 = torch.cat((y44, y3), dim = 1)
        y33 = self.ccu2(y33)

        y22 = torch.cat((y33, y2), dim = 1)
        y22 = self.ccu3(y22)

        y11 = torch.cat((y22, y1), dim = 1)
        y11 = self.conv3(y11)
        y11 = self.conv1(y11)

        return y11

