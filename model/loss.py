import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        #print(inputs.squeeze())
        #print(targets.float())
        targets = targets.float()
        p = torch.sigmoid(inputs.squeeze())
        ce_loss = F.binary_cross_entropy(p,  targets)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t)**self.gamma * ce_loss
        loss = loss.mean()
        return loss
        
class BCELoss(nn.Module):
    def __init__(self,pos_weight=2):
        super(BCELoss, self).__init__()
        self.pos_weight = torch.tensor(pos_weight).cuda() 

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        targets = targets.float()
        p = torch.sigmoid(inputs.squeeze())
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = criterion(p,  targets)
        return loss

def _expand_binary_labels(labels, probs, label_channels=2):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    for idx, l in enumerate(labels):
        bin_labels[idx, 0] = l
        bin_labels[idx, 1] = 1 - l
    bin_label_weights = torch.ones(labels.size(0), label_channels)
    probs=probs.view(-1, 1)
    bin_probs = probs.new_full((probs.size(0), label_channels), 0)
    for idx, p in enumerate(probs):
        bin_probs[idx, 0] = p
        bin_probs[idx, 1] = 1 - p
    return bin_labels, bin_probs, bin_label_weights

class GHMC_loss(nn.Module):
    def __init__(self, bins=30, momentum=0.75, loss_weight=1.0):
        super(GHMC_loss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        '''
        :param pred:[batch_num, class_num]:
        :param target:[batch_num, class_num]:Binary class target for each sample.
        :param label_weight:[batch_num, class_num]: the value is 1 if the sample is valid and 0 if ignored.
        :return: GHMC_Loss
        '''

        prob = torch.sigmoid(pred.squeeze())
        target, pred, label_weight = _expand_binary_labels(target.float(), prob)
        edges = torch.Tensor(self.edges).float().cuda()
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred - target.float())
        valid = label_weight > 0
        total = max(valid.float().sum().item(), 1.0)
        n = 0  # the number of valid bins

        for i in range(self.bins):
            inds = (g >= edges[i]) & (g <= edges[i + 1])
            num_in_bins = inds.sum().item()
            if num_in_bins > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bins
                    weights[inds] = total / self.acc_sum[i]
                else:
                    weights[inds] = total / num_in_bins
                n += 1

        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction='sum') / total

        return loss * self.loss_weight


