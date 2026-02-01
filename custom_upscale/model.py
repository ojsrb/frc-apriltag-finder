import math

import torch
import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self, in_channels, channels, scale_factor):
        super(ESPCN, self).__init__()
        hidden_channels = channels // 2
        out_channels = int(in_channels * (scale_factor ** 2))

        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, (5,5), (1,1), (2,2)),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, (3,3), (1,1), (1,1)),
            nn.Tanh(),
        )

        self.sub_pixel = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, (3,3), (1,1), (1,1)),
            nn.PixelShuffle(scale_factor),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                    nn.init.zeros_(module.bias.data)


    def forward(self, x):
        return self._forward_impl(x)

    def _forward_impl(self, x):
        x = self.feature_maps(x)
        x = self.sub_pixel(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x