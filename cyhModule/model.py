from torch import Tensor, nn
import torch
from torch.nn import functional as F

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, model):
        super(LinearLayer, self).__init__()
        if 'ViT' in model:
            self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(9)])
        else:
            self.fc = nn.ModuleList([nn.Linear(dim_in * 2 ** (i + 2), dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:  # NLD
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens


class LinearLayerTuning(nn.Module):
    def __init__(self):
        super().__init__()

        # self.fc = nn.Linear(768, 768) # 14
        # self.fc = nn.Linear(512, 512)
        self.fc = nn.Linear(640, 640) # v16+
        self.act = nn.GELU()

        # self.fc2 = nn.Linear(512, 512)
        # self.act2 = nn.GELU()

        self.initial_paras()

    def initial_paras(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.002)

    def forward(self, tokens):
        tokens = self.fc(tokens)
        tokens = self.act(tokens)

        return tokens

