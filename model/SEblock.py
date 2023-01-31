import torch.nn as nn

class SENet(nn.Module):
    def __init__(self, in_channels, reduction = 4):
        super(SENet, self).__init__()
        self.chans = in_channels
        self.mid_chans = self.chans // reduction

        self.SEblock = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=self.chans, out_channels=self.mid_chans, 
                        kernel_size=1, stride=1, groups=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.mid_chans, out_channels=self.chans, 
                        kernel_size=1, stride=1, groups=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.SEblock(x)
        x = x * w
        return x
