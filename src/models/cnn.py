''' File made to test data logic for CNN ML training model
Iterates over the Radar image using different resolutions to learn the features that are in Non-tornadic versus Tornadic cells '''


# Importing packages
import torch 
import torch.nn as nn

class TornadoCNN (nn.Module):
    def __init__(self, in_channels=13):
        super(TornadoCNN, self).__init__()

        self.features = nn.Sequential(
            #Block 1, earlier scan (32, 60, 120)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            #Block 2, middle scan (64, 30, 60)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            #Block 3, last scan (128, 15, 30)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1)) # (128, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            )
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x