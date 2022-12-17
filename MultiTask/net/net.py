#LeNet
# import the necessary packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn as nn
from collections import OrderedDict

class LeNet(Module):
    def __init__(self, numChannels, classes):
        
        # call the parent constructor
    		super(LeNet, self).__init__()
            
    		# initialize first set of CONV => RELU => POOL layers
    		self.conv1 = Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5, 5))
    		self.relu1 = ReLU()
    		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            
    		# initialize second set of CONV => RELU => POOL layers
    		self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
    		self.relu2 = ReLU()
    		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            
    		# initialize first (and only) set of FC => RELU layers
    		self.fc1 = Linear(in_features=800, out_features=500)
    		self.relu3 = ReLU()
            
    		# initialize our softmax classifier
    		self.fc2 = Linear(in_features=500, out_features=classes)
    		self.logSoftmax = LogSoftmax(dim=1)
        
    def forward(self, x):
    		x = self.conv1(x)
    		x = self.relu1(x)
    		x = self.maxpool1(x)
        
    		# pass the output from the previous layer through the second
    		# set of CONV => RELU => POOL layers
    		x = self.conv2(x)
    		x = self.relu2(x)
    		x = self.maxpool2(x)
            
    		# flatten the output from the previous layer and pass it
    		# through our only set of FC => RELU layers
    		x = flatten(x, 1)
    		x = self.fc1(x)
    		x = self.relu3(x)
            
    		# pass the output to our softmax classifier to get our output
    		# predictions
    		x = self.fc2(x)
    		output = self.logSoftmax(x)
            
    		# return the output predictions
    		return output
        
        
class VGG(nn.Module):
    """
    Standard PyTorch implementation of VGG. Pretrained imagenet model is used.
    """
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

        # We need these for MaxUnpool operation
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        self.feature_maps = OrderedDict()
        self.pool_locs = OrderedDict()
        
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x