import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, chn, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(chn, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(chn=3):
    return ResNet(chn,BasicBlock, [2, 2, 2, 2])


class MLP(nn.Module):
    def __init__(self, idim=28*28, hdim=500, odim=10, init=False, \
                    activation=nn.LeakyReLU, \
                    use_bn=False, input_dropout=0, dropout=0, bias=False):
        super().__init__()
        self.idim = idim
        self.hdim = hdim 
        self.odim = odim 

        self.fc1 = nn.Linear(idim, hdim)
        #if input_dropout > 0: layers.append(nn.Dropout(input_dropout))
        self.activation = activation()
        self.fc2 = nn.Linear(hdim, odim, bias=bias)
        #if use_dropout: layers.append(nn.Dropout(dropout))
        #if use_bn: layers.append(nn.BatchNorm1d(hdim))
        
        if init: 
            self.apply(self._kaiming_init)
    
    def _kaiming_init(self):
        if isinstance(self, nn.Linear):
            nn.init.kaiming_uniform_(self.weight.data)
            nn.init.kaiming_uniform_(self.bias.data)
    
    def forward(self, x):
        x = x.view(-1, self.idim)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x 


class MLP2(nn.Module):
    def __init__(self, idim=28*28, hdim=512, odim=10, init=False, \
                    activation=nn.LeakyReLU, \
                    use_bn=False, input_dropout=0, dropout=0, bias=False):
        super().__init__()
        self.idim = idim
        self.hdim = hdim 
        self.odim = odim 

        self.fc1 = nn.Linear(idim, hdim)
        #if input_dropout > 0: layers.append(nn.Dropout(input_dropout))
        self.activation = activation()
        self.fc2 = nn.Linear(hdim, 128)
        self.activation2 = activation()
        self.fc3 = nn.Linear(128, odim, bias=bias)
        #if use_dropout: layers.append(nn.Dropout(dropout))
        #if use_bn: layers.append(nn.BatchNorm1d(hdim))
        
        if init: 
            self.apply(self._kaiming_init)
    
    def _kaiming_init(self):
        if isinstance(self, nn.Linear):
            nn.init.kaiming_uniform_(self.weight.data)
            nn.init.kaiming_uniform_(self.bias.data)
    
    def forward(self, x):
        x = x.view(-1, self.idim)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)
        return x 

class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes=10):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        #self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        #self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.act1 = nn.LeakyReLU()
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        #self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(4608, 128)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.act1(out)
        #out = self.max_pool1(out)
        
        out = self.conv_layer2(out)
        #out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
