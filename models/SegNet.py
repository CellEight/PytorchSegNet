import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__() 
        # Encoder Layers
        self.conv1_1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv2_1 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv3_1 = nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv3_2 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv4_1 = nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv4_2 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv5_1 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv5_2 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.pool  = nn.MaxPool2d(2,2, return_indices=True)
        # Decoder Layers
        self.unpool = nn.MaxUnpool2d(2,2) 
        self.conv6_1 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv6_2 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv7_1 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv7_2 = nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv8_1 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv8_2 = nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv9_1 = nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.conv10_1 = nn.Conv2d(64,n_classes,kernel_size=3,stride=1,padding=1,padding_mode='reflect') 
        self.applyPretraining()

    def forward(self, x):
        # Encode
        x = F.relu(self.conv1_1(x))
        x, ind1 = self.pool(x)
        x = F.relu(self.conv2_1(x))
        x, ind2 = self.pool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x, ind3 = self.pool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x, ind4 = self.pool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x, ind5 = self.pool(x)
        # Decode
        x = self.unpool(x,id5)
        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        x = self.unpool(x,id4)
        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))
        x = self.unpool(x,id3)
        x = F.relu(self.conv8_1(x))
        x = F.relu(self.conv8_2(x))
        x = self.unpool(x,ind2)
        x = F.relu(self.conv9_1(x))
        x = self.unpool(x,ind1)
        x = F.relu(self.conv10_1(x))
        return x

    def applyPretraining(self):
        with torch.no_grad():
            vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
            param = list(vgg16.features.parameters())
            names = ['conv1_1.weight', 'conv1_1.bias', 'conv2_1.weight', 'conv2_1.bias', 'conv3_1.weight', 'conv3_1.bias', 'conv3_2.weight', 'conv3_2.bias', 'conv4_1.weight', 'conv4_1.bias', 'conv4_2.weight', 'conv4_2.bias', 'conv5_1.weight', 'conv5_1.bias', 'conv5_2.weight', 'conv5_2.bias']
            for name, state in zip(names, vgg16.features.state_dict()):
                self.state_dict()[name] = state
    
