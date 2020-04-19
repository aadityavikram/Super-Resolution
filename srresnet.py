import torch.nn as nn


class SRResNet(nn.Module):
    def __init__(self):
        super(SRResNet, self).__init__()

        # Scaling factor must be 2, 4, or 8 and half of desired upscaling
        scaling_factor = 2

        # The first convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=9 // 2)
        self.activation1 = nn.PReLU()

        # A sequence of n_blocks residual blocks, each containing a skip-connection across the block
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm2 = nn.BatchNorm2d(num_features=64)
        self.activation2 = nn.PReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm3 = nn.BatchNorm2d(num_features=64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm4 = nn.BatchNorm2d(num_features=64)
        self.activation4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm5 = nn.BatchNorm2d(num_features=64)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm6 = nn.BatchNorm2d(num_features=64)
        self.activation6 = nn.PReLU()

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm7 = nn.BatchNorm2d(num_features=64)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm8 = nn.BatchNorm2d(num_features=64)
        self.activation8 = nn.PReLU()

        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm9 = nn.BatchNorm2d(num_features=64)

        self.conv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm10 = nn.BatchNorm2d(num_features=64)
        self.activation10 = nn.PReLU()

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm11 = nn.BatchNorm2d(num_features=64)

        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm12 = nn.BatchNorm2d(num_features=64)
        self.activation12 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm13 = nn.BatchNorm2d(num_features=64)

        self.conv14 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm14 = nn.BatchNorm2d(num_features=64)
        self.activation14 = nn.PReLU()

        self.conv15 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm15 = nn.BatchNorm2d(num_features=64)

        self.conv16 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm16 = nn.BatchNorm2d(num_features=64)
        self.activation16 = nn.PReLU()

        self.conv17 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm17 = nn.BatchNorm2d(num_features=64)

        self.conv18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm18 = nn.BatchNorm2d(num_features=64)
        self.activation18 = nn.PReLU()

        self.conv19 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm19 = nn.BatchNorm2d(num_features=64)

        self.conv20 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm20 = nn.BatchNorm2d(num_features=64)
        self.activation20 = nn.PReLU()

        self.conv21 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm21 = nn.BatchNorm2d(num_features=64)

        self.conv22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm22 = nn.BatchNorm2d(num_features=64)
        self.activation22 = nn.PReLU()

        self.conv23 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm23 = nn.BatchNorm2d(num_features=64)

        self.conv24 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm24 = nn.BatchNorm2d(num_features=64)
        self.activation24 = nn.PReLU()

        self.conv25 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm25 = nn.BatchNorm2d(num_features=64)

        self.conv26 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm26 = nn.BatchNorm2d(num_features=64)
        self.activation26 = nn.PReLU()

        self.conv27 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm27 = nn.BatchNorm2d(num_features=64)

        self.conv28 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm28 = nn.BatchNorm2d(num_features=64)
        self.activation28 = nn.PReLU()

        self.conv29 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm29 = nn.BatchNorm2d(num_features=64)

        self.conv30 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm30 = nn.BatchNorm2d(num_features=64)
        self.activation30 = nn.PReLU()

        self.conv31 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm31 = nn.BatchNorm2d(num_features=64)

        self.conv32 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm32 = nn.BatchNorm2d(num_features=64)
        self.activation32 = nn.PReLU()

        self.conv33 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm33 = nn.BatchNorm2d(num_features=64)

        # Another convolutional block
        self.conv34 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.norm34 = nn.BatchNorm2d(num_features=64)

        # Upscaling is done by sub-pixel convolution, with each such block upscaling by a factor of 2 (total 4 upscale)
        self.conv35 = nn.Conv2d(in_channels=64, out_channels=64 * (scaling_factor ** 2), kernel_size=3, padding=3 // 2)
        self.pixel_shuffle35 = nn.PixelShuffle(upscale_factor=2)
        self.activation35 = nn.PReLU()

        self.conv36 = nn.Conv2d(in_channels=64, out_channels=64 * (scaling_factor ** 2), kernel_size=3, padding=3 // 2)
        self.pixel_shuffle36 = nn.PixelShuffle(upscale_factor=2)
        self.activation36 = nn.PReLU()

        # The last convolutional block
        self.conv37 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=9 // 2)
        self.activation37 = nn.Tanh()

    def forward(self, x):
        # convolutional block (1)
        x = self.conv1(x)
        x = self.activation1(x)

        res_init = x
        res = x

        # residual blocks (16)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = x + res
        res = x

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.activation4(x)

        x = self.conv5(x)
        x = self.norm5(x)

        x = x + res
        res = x

        x = self.conv6(x)
        x = self.norm6(x)
        x = self.activation6(x)

        x = self.conv7(x)
        x = self.norm7(x)

        x = x + res
        res = x

        x = self.conv8(x)
        x = self.norm8(x)
        x = self.activation8(x)

        x = self.conv9(x)
        x = self.norm9(x)

        x = x + res
        res = x

        x = self.conv10(x)
        x = self.norm10(x)
        x = self.activation10(x)

        x = self.conv11(x)
        x = self.norm11(x)

        x = x + res
        res = x

        x = self.conv12(x)
        x = self.norm12(x)
        x = self.activation12(x)

        x = self.conv13(x)
        x = self.norm13(x)

        x = x + res
        res = x

        x = self.conv14(x)
        x = self.norm14(x)
        x = self.activation14(x)

        x = self.conv15(x)
        x = self.norm15(x)

        x = x + res
        res = x

        x = self.conv16(x)
        x = self.norm16(x)
        x = self.activation16(x)

        x = self.conv17(x)
        x = self.norm17(x)

        x = x + res
        res = x

        x = self.conv18(x)
        x = self.norm18(x)
        x = self.activation18(x)

        x = self.conv19(x)
        x = self.norm19(x)

        x = x + res
        res = x

        x = self.conv20(x)
        x = self.norm20(x)
        x = self.activation20(x)

        x = self.conv21(x)
        x = self.norm21(x)

        x = x + res
        res = x

        x = self.conv22(x)
        x = self.norm22(x)
        x = self.activation22(x)

        x = self.conv23(x)
        x = self.norm23(x)

        x = x + res
        res = x

        x = self.conv24(x)
        x = self.norm24(x)
        x = self.activation24(x)

        x = self.conv25(x)
        x = self.norm25(x)

        x = x + res
        res = x

        x = self.conv26(x)
        x = self.norm26(x)
        x = self.activation26(x)

        x = self.conv27(x)
        x = self.norm27(x)

        x = x + res
        res = x

        x = self.conv28(x)
        x = self.norm28(x)
        x = self.activation28(x)

        x = self.conv29(x)
        x = self.norm29(x)

        x = x + res
        res = x

        x = self.conv30(x)
        x = self.norm30(x)
        x = self.activation30(x)

        x = self.conv31(x)
        x = self.norm31(x)

        x = x + res
        res = x

        x = self.conv32(x)
        x = self.norm32(x)
        x = self.activation32(x)

        x = self.conv33(x)
        x = self.norm33(x)

        x = x + res

        # convolutional block (1)
        x = self.conv34(x)
        x = self.norm34(x)

        x = x + res_init  # adding skip connection

        # sub-pixel convolutional blocks (2)
        x = self.conv35(x)
        x = self.pixel_shuffle35(x)
        x = self.activation35(x)

        x = self.conv36(x)
        x = self.pixel_shuffle36(x)
        x = self.activation36(x)

        # convolutional block (1)
        x = self.conv37(x)
        x = self.activation37(x)

        return x
