import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from models.utils import one_hot_embedding


class FocalLoss(nn.Module):
    def __init__(self, num_classes=1):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu().long(), 1 + self.num_classes)  # [N, 81]
        t = t[:, 1:]  # exclude background
        t = Variable(t).cuda()

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)
        w = alpha * t + (1 - alpha) * (1 - t)
        w = w * (1 - pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y):
        alpha = 0.25
        
        t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)
        t = t[:, 1:]
        t = Variable(t).cuda()

        xt = x * (2 * t - 1)
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()

    @staticmethod
    def where(cond, x_1, x_2):
        return (cond.float() * x_1) + ((1 - cond.float()) * x_2)

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()
        num_pos = num_pos.type(torch.cuda.FloatTensor)

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)
        masked_loc_preds = loc_preds[mask].view(-1, 4)
        masked_loc_targets = loc_targets[mask].view(-1, 4)
        # loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
        regression_diff = torch.abs(masked_loc_targets - masked_loc_preds)
        loc_loss = self.where(torch.le(regression_diff, 1.0 / 9.0), 0.5 * 9.0 * torch.pow(regression_diff, 2), regression_diff - 0.5 / 9.0)
        loc_loss = loc_loss.mean()

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        # loc_loss = loc_loss / num_pos
        loc_loss = loc_loss
        cls_loss = cls_loss / num_pos

        return loc_loss, cls_loss