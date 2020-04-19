import torch
import torch.nn as nn
from super_resolution_new.srresnet import SRResNet


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = SRResNet()

    def load_model(self):
        ckpt = torch.load('model/srresnet_14.pt')
        self.net.load_state_dict(ckpt['state_dict'])

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.activation1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=3 // 2)
        self.norm2 = nn.BatchNorm2d(num_features=64)
        self.activation2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3 // 2)
        self.norm3 = nn.BatchNorm2d(num_features=128)
        self.activation3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=3 // 2)
        self.norm4 = nn.BatchNorm2d(num_features=128)
        self.activation4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3 // 2)
        self.norm5 = nn.BatchNorm2d(num_features=256)
        self.activation5 = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=3 // 2)
        self.norm6 = nn.BatchNorm2d(num_features=256)
        self.activation6 = nn.LeakyReLU(0.2)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2)
        self.norm7 = nn.BatchNorm2d(num_features=512)
        self.activation7 = nn.LeakyReLU(0.2)

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=3 // 2)
        self.norm8 = nn.BatchNorm2d(num_features=512)
        self.activation8 = nn.LeakyReLU(0.2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation2(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.activation3(out)

        out = self.conv4(out)
        out = self.norm4(out)
        out = self.activation4(out)

        out = self.conv5(out)
        out = self.norm5(out)
        out = self.activation5(out)

        out = self.conv6(out)
        out = self.norm6(out)
        out = self.activation6(out)

        out = self.conv7(out)
        out = self.norm7(out)
        out = self.activation7(out)

        out = self.conv8(out)
        out = self.norm8(out)
        out = self.activation8(out)

        out = self.avgpool(out)
        out = self.fc1(out.view(x.size(0), -1))  # (batch_size, -1)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        return out
