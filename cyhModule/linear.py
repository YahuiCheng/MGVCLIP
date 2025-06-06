import torch
import torch.nn as nn

import numpy as np


class Text_prompt_layer(nn.Module):
    def __init__(self):
        super(Text_prompt_layer, self).__init__()
        self.fc = nn.Linear(226, 77) # v16+
        self.act = nn.GELU()
        self.initial_paras()

    def initial_paras(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, x):
        image_prompt = 0
        for xi in x:
            xi = torch.max(xi, dim=2)[0]
            image_prompt += xi
        image_prompt = self.fc(image_prompt)
        image_prompt = self.act(image_prompt)

        return image_prompt



