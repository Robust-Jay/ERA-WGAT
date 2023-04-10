import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision.models import vgg19


class CompoundLoss(_Loss):
    def __init__(self, weight=0.1):
        super().__init__()
        self.__weight = weight
        vgg = vgg19(pretrained=True)
        feature_extractor = list(vgg.features.children())[:35]
        self.__blocks = nn.ModuleList()
        self.__blocks.append(nn.Sequential(*feature_extractor[:8]))
        self.__blocks.append(nn.Sequential(*feature_extractor[8:17]))
        self.__blocks.append(nn.Sequential(*feature_extractor[17:25]))
        self.__blocks.append(nn.Sequential(*feature_extractor[25:]))
        self.__criterion = nn.MSELoss()

    def forward(self, X, Y):
        mse_loss = self.__criterion(X, Y)

        if X.shape[1] != 3:
            X = X.repeat(1, 3, 1, 1)
            Y = Y.repeat(1, 3, 1, 1)

        input_feats = []
        target_feats = []

        for block in self.__blocks:
            X = block(X)
            Y = block(Y)
            input_feats.append(X)
            target_feats.append(Y)

        vgg_loss = 0.
        for i in range(len(input_feats)):
            vgg_loss += self.__criterion(input_feats[i], target_feats[i])

        vgg_loss /= len(input_feats)
        loss = mse_loss + self.__weight * vgg_loss
        return loss


if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.randn(2, 1, 512, 512)
    y = torch.randn(2, 1, 512, 512)
    x = x.to(device)
    y = y.to(device)

    loss = CompoundLoss().to(device)
    loss(x, y)
    # macs, params = profile(model, inputs=(input,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print()
