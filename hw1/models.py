import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            )
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=1,
                padding=0,
                bias=False,
            )
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
            )
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=1,
                padding=0,
                bias=False,
            )
        self.relu6 = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(
            in_features=512,
            out_features=256,
        )

        self.relu7 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(
            in_features=256,
            out_features=100,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.relu6(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu7(x)
        x = self.fc2(x)

        return x
