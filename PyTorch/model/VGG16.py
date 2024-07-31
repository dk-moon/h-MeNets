import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class VGG16(nn.Module):
    def __init__(self, hidden_features=4096, out_features=2):
        super(VGG16, self).__init__()

        self.hidden_features=hidden_features
        self.out_features=out_features
        self.p = hidden_features + 1
        
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.convNet = vgg16.features        
        # replace the maxpooling layer in VGG
        self.convNet[4] = nn.MaxPool2d(kernel_size=2, stride=1)
        self.convNet[9] = nn.MaxPool2d(kernel_size=2, stride=1)

        # MLP
        self.FC = nn.Sequential(
            nn.Linear(512*4*7, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_features+2, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Linear(hidden_features + 1, out_features, bias=False)



    def get_feature_map(self, image, head_pose) : 
        feature = self.convNet(image)
        feature = torch.flatten(feature, start_dim=1)
        feature = self.FC(feature)
        feature = self.fc1(torch.cat([feature, head_pose], dim=1))
        return torch.cat([torch.ones_like(feature[:,0]).unsqueeze(1), feature], dim=1)
      

    def forward(self, image, head_pose):
        return self.fc2(self.get_feature_map(image, head_pose))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)