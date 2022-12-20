import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential (
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, i):
    
        identity = i

        out = self.conv1(identity)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)

        #Reason for downsample: the convolutional layer outputs a tensor with shape (4, 128, 192, 192) [FIRST CASE] with 128 being the output channels sent as input. 
        #[FIRST CASE] the input has shape (4, 64, 192, 192), so to match the two to perform the skip connection it is necessary to reshape the output of the previous layers as to match the input
        identity = self.downsample(identity)

        out += identity
        out = self.relu2(out)

        return out
        
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

    def get_last_shared_layer(self):
        return self.double_conv[len(self.double_conv)-1]


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #DoubleConv(in_channels, out_channels)
            BasicBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

    def get_last_shared_layer(self):
        return self.maxpool_conv[len(self.maxpool_conv)-1].get_last_shared_layer()


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        self.ann1_1 = nn.Linear(294912, 64)
        self.ann1_2 = nn.Linear(64,7)
        self.ann2_1 = nn.Linear(294912,64)
        self.ann2_2 = nn.Linear(64,1)
        
        self.weights = torch.nn.Parameter(torch.ones(3).float())

    def forward(self, x):
        # first layer
        x1 = self.inc(x)
        # start encoder part
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #decoder part
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # segmentation output
        output = self.outc(x)

        # flatten of the output of the encoder
        a = x5.view(x5.size(0), -1) # embedding layer

        # label classification part
        a1 = self.ann1_1(a)
        a1 = F.relu(a1)
        a1 = self.ann1_2(a1)
        lab = a1 # label classification output - multi

        # intensity classification part
        a2 = self.ann2_1(a)
        a2 = F.relu(a2)        
        a2 = self.ann2_2(a2)
        intensity = a2 # intensity classification ouput - binary

        return [output, lab, intensity]

    def get_last_shared_layer(self):
        return self.down4

