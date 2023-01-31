import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet169(pretrained=True)
        self.fc1 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 179)
        )

    def forward(self, x):
        out = self.densenet(x)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return F.log_softmax(out)
        