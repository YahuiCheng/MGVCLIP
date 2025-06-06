import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss


class Cross_Entropy_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, targets):

        gt = torch.zeros_like(input)
        gt.scatter_(1, targets.view(-1, 1).cuda(), 1)

        loss = -gt * torch.log(input)  # 以e为底
        loss = torch.mean(torch.sum(loss, dim=1))

        return loss


# class Discriminability_prompt_loss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, text_feature):
#         text_feature1, text_feature2 = text_feature
#         # cos = torch.dot(text_feature1, text_feature2)
#         # loss = (1 + cos)**2 / 2     # v1 负相关
#         # loss = (torch.exp(cos) - 1)**2 / 2    # v2 正交
#         loss = torch.exp(-torch.sum((text_feature1 - text_feature2)**2) / 2)   # v3 最大化欧式距离
#
#         text_feature1 = text_feature1 / text_feature1.norm(dim=-1, keepdim=True)
#         cos1 = text_feature1 @ text_feature1.t()
#         cos1 = torch.mean(cos1)
#         text_feature2 = text_feature2 / text_feature2.norm(dim=-1, keepdim=True)
#         cos2 = text_feature2 @ text_feature2.t()
#         cos2 = torch.mean(cos2)
#         cos_dis = torch.exp(-(cos1 + cos2) / 2)
#         loss = loss + cos_dis
#         return loss

# v1
class Discriminability_prompt_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, text_feature):
        text_feature1, text_feature2 = text_feature
        # text_feature1 = torch.mean(text_feature1, dim=1)
        # text_feature2 = torch.mean(text_feature2, dim=1)
        # cos = torch.dot(text_feature1, text_feature2)
        # loss = (1 + cos)**2 / 2     # v1 负相关
        # loss = (torch.exp(cos) - 1)**2 / 2    # v2 正交
        loss = torch.exp(-torch.sum((text_feature1 - text_feature2)**2) / 2)   # v3 最大化欧式距离
        return loss

class Image_prompt_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prompt, image_prompt, gt):
        prompt = torch.mean(prompt, dim=2)
        image_prompt = torch.squeeze(image_prompt, 2)

        prompt = prompt / prompt.norm(dim=-1, keepdim=True)
        image_prompt = image_prompt / image_prompt.norm(dim=-1, keepdim=True)

        cos = image_prompt @ prompt.t()

        # cos = torch.dot(text_feature1, text_feature2)
        # loss = (torch.exp(cos) - 1)**2 / 2 # 正交、

        return cos

class Variance_Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = 240
        self.h = 240

    def forward(self, pred_mask, gt_mask):
        loss = 0
        for wi in range(0, self.w, 20):
            for hi in range(0, self.h, 20):
                # E(x^2) - E^2(x)
                pred_patch = pred_mask[:, wi:wi+20, hi:hi+20]
                pred_patch_mean = torch.mean(pred_mask)

                pred_patch_var = torch.mean((pred_patch - pred_patch_mean)**2)

                gt_patch = gt_mask[wi:wi+20, hi:hi+20]
                gt_patch_mean = torch.mean(pred_mask)
                gt_patch_var = torch.mean((gt_patch - gt_patch_mean)**2)

                loss += (pred_patch_var - gt_patch_var)**2 / 2

        return loss / 12

if __name__ == "__main__":

    a = torch.tensor([5., 6., 16., 9.])
    mean = torch.mean(a)
    print(mean)
    var = torch.mean((a - mean)**2)
    print(var)
