import torch
import torch.nn as nn

class LeNet(nn.Module) :
    def __init__(self, hidden_features = 500, out_features = 2) :
        super(LeNet, self).__init__()

        self.hidden_features = hidden_features
        self.out_features = out_features
        self.model_name = "LeNet-5"
        self.p = hidden_features + 3

        self.lenet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=50*6*12, out_features=hidden_features),
            nn.LeakyReLU()
        )
        self.fc = nn.Linear(in_features=self.hidden_features + 2, out_features=out_features)

    def forward(self, image, head_pose) :
        x = image.unsqueeze(1)
        x = self.lenet(x)
        x = torch.cat([x, head_pose], dim = 1)
        x = self.fc(x)
        return x