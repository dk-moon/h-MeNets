import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet_batchnorm(nn.Module) : 
    def __init__(self, hidden_features = 500, out_features = 1) : 
        super(ResNet_batchnorm, self).__init__()

        self.hidden_features = hidden_features
        self.out_features = out_features
        self.model_name = "ResNet-18 (batchnorm)"
        self.p = hidden_features + 3

        # conv1 : data -> conv1 -> pool1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=2, bias=False), 
            nn.BatchNorm2d(num_features=64), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # pool1 -> res2a_branch1
        self.res2a_branch1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False), 
            nn.BatchNorm2d(num_features=64)
        )
        # pool1 -> res2a_brach2a -> res2a_brach2b
        self.res2a_branch2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.BatchNorm2d(num_features=64), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.BatchNorm2d(num_features=64)
        )
        # res2a = Relu(branch1 + branch2)
        # res2a -> res2b_branch2a -> res2b_branch2b
        self.res2b_branch2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.BatchNorm2d(num_features=64), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.BatchNorm2d(num_features=64)
        )
        # res2b = ReLU(res2a + res_branch2b)
        # res2b -> res3a_branch1
        self.res3a_branch1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=2, bias=False), 
            nn.BatchNorm2d(num_features=128)
        )

        # res2b -> res3a_branch2a -> res3a_branch2b
        self.res3a_branch2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2, bias=False), 
            nn.BatchNorm2d(num_features=128), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.BatchNorm2d(num_features=128)
        )
        # res3a = ReLU(res3a_branch1 + res3a_branch2)
        # res3a -> res3b_branch2a -> res3b_branch2b
        self.res3b_branch2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.BatchNorm2d(num_features=128), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.BatchNorm2d(num_features=128)
        )
        # res3b = ReLU(res3a + res3b_branch2b)
        # res3b -> res4a_branch1
        self.res4a_branch1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0, stride=2, bias=False), 
            nn.BatchNorm2d(num_features=256)
        )
        # res3b -> res4a_branch2a -> res4a_branch2b
        self.res4a_branch2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2, bias=False), 
            nn.BatchNorm2d(num_features=256), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.BatchNorm2d(num_features=256)
        )
        # res4a = ReLU(res4a_branch1 + res4a_branch2b)
        # res4a -> res4b_branch2a -> res4b_branch2b
        self.res4b_branch2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=256), 
            nn.ReLU(),  
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.BatchNorm2d(num_features=256)
        )
        # res4b = ReLU(res4a + res4b_branch2b)
        # res4b -> res5a_branch1
        self.res5a_branch1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0, stride=2, bias=False), 
            nn.BatchNorm2d(num_features=512)
        )
        # res4b -> res5a_branch2a -> res5a_branch2b
        self.res5a_branch2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=512), 
            nn.ReLU(),  
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.BatchNorm2d(num_features=512)
        )
        # res5a = ReLU(res5a_branch1 + res5a_branch2b)
        # res5a -> res5b_branch2a -> res5b_branch2b
        self.res5b_branch2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=512), 
            nn.ReLU(),  
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.BatchNorm2d(num_features=512)
        )
        # res5b = ReLU(res5a + res5b_branch2b)
        # res5b -> pool5
        # self.pseudo_pool5 = nn.AvgPool2d(kernel_size=(1,2))
        self.pool5 = nn.AdaptiveAvgPool2d((1,1))

        # pool5 -> ip1
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=hidden_features), 
            nn.BatchNorm1d(num_features=hidden_features),
            nn.ReLU(),
            # nn.Dropout(p = 0.7),
        )
        # cat = concat(ip1, headpose)
        self.fc2 = nn.Linear(in_features=hidden_features + 2 + 1, out_features=out_features, bias = False)

    def get_feature_map(self, image, head_pose) : 
        '''
        image    : [B x 60 x 36]
        headpose : [B x  2]
        Output   : [B x (503)]
        '''

        # Assume that data :                                        # [B x     x 36 x 60]
        x = image.unsqueeze(1)
        # Unsqueezed data  :                                        # [B x   1 x 36 x 60]
        # data -> pool1 
        x = self.conv1(x)                                           # [B x  64 x  8 x 14]
        # pool1 -> res2a
        x = F.relu(self.res2a_branch1(x) + self.res2a_branch2(x))   # [B x  64 x  8 x 14]
        # res2a -> res2b
        x = F.relu(x + self.res2b_branch2(x))                       # [B x 128 x  8 x 14]
        # res2b -> res3a
        x = F.relu(self.res3a_branch1(x) + self.res3a_branch2(x))   # [B x 128 x  4 x  7]
        # res3a -> res3b
        x = F.relu(x + self.res3b_branch2(x))                       # [B x 256 x  4 x  7]
        # res3b -> res4a
        x = F.relu(self.res4a_branch1(x) + self.res4a_branch2(x))   # [B x 256 x  2 x  4]
        # res4a -> res4b
        x = F.relu(x + self.res4b_branch2(x))                       # [B x 256 x  2 x  4]
        # res4b -> res5a
        x = F.relu(self.res5a_branch1(x) + self.res5a_branch2(x))   # [B x 512 x  1 x  2]
        # res5a -> res5
        x = F.relu(x + self.res5b_branch2(x))                       # [B x 512 x  1 x  2]
        # res5b -> pool5
        x = self.pool5(x).flatten(1)                                # [B x 512]
        # pool5 -> ip1
        x = self.fc1(x)                                             # [B x 500]
        # (ip1, headpose) -> concatenation
        output = torch.cat([
            torch.ones_like(x[:,0]).unsqueeze(1),                   # [B x 1]
            x, head_pose                                            # [B x 2]
        ], dim=1)                                                   # [B x 503]
        return output

    def forward(self, image, head_pose) : 
        x = self.get_feature_map(image, head_pose)                   # [B x 503]
        x = self.fc2(x)
        if self.out_features == 1 : 
            return x.squeeze(1)
        else : 
            return x