import torch
import numpy as np

class PearsonR(torch.nn.Module):
    def __init__(self, num_targets, summarize=True):
        super(PearsonR, self).__init__()
        self.summarize = summarize
        self.register_buffer('count', torch.zeros(num_targets))
        self.register_buffer('product', torch.zeros(num_targets))
        self.register_buffer('true_sum', torch.zeros(num_targets))
        self.register_buffer('true_sumsq', torch.zeros(num_targets))
        self.register_buffer('pred_sum', torch.zeros(num_targets))
        self.register_buffer('pred_sumsq', torch.zeros(num_targets))

    def forward(self, y_true, y_pred):
        product = torch.sum(y_true * y_pred, dim=0)
        true_sum = torch.sum(y_true, dim=0)
        true_sumsq = torch.sum(y_true ** 2, dim=0)
        pred_sum = torch.sum(y_pred, dim=0)
        pred_sumsq = torch.sum(y_pred ** 2, dim=0)
        count = torch.ones_like(y_true).sum(dim=0)

        self.product += product
        self.true_sum += true_sum
        self.true_sumsq += true_sumsq
        self.pred_sum += pred_sum
        self.pred_sumsq += pred_sumsq
        self.count += count

        true_mean = self.true_sum / self.count
        pred_mean = self.pred_sum / self.count

        covariance = self.product - true_mean * self.pred_sum - pred_mean * self.true_sum + self.count * true_mean * pred_mean
        true_var = self.true_sumsq - self.count * true_mean ** 2
        pred_var = self.pred_sumsq - self.count * pred_mean ** 2

        denominator = torch.sqrt(true_var * pred_var)
        correlation = covariance / denominator

        if self.summarize:
            return correlation.mean()
        else:
            return correlation

    def reset(self):
        self.count.zero_()
        self.product.zero_()
        self.true_sum.zero_()
        self.true_sumsq.zero_()
        self.pred_sum.zero_()
        self.pred_sumsq.zero_()


class R2(torch.nn.Module):
    def __init__(self, num_targets, summarize=True):
        super(R2, self).__init__()
        self.summarize = summarize
        self.register_buffer('count', torch.zeros(num_targets))
        self.register_buffer('true_sum', torch.zeros(num_targets))
        self.register_buffer('true_sumsq', torch.zeros(num_targets))
        self.register_buffer('product', torch.zeros(num_targets))
        self.register_buffer('pred_sumsq', torch.zeros(num_targets))

    def forward(self, y_true, y_pred):
        true_sum = torch.sum(y_true, dim=0)
        true_sumsq = torch.sum(y_true ** 2, dim=0)
        product = torch.sum(y_true * y_pred, dim=0)
        pred_sumsq = torch.sum(y_pred ** 2, dim=0)
        count = torch.ones_like(y_true).sum(dim=0)

        self.true_sum += true_sum
        self.true_sumsq += true_sumsq
        self.product += product
        self.pred_sumsq += pred_sumsq
        self.count += count

        true_mean = self.true_sum / self.count
        total = self.true_sumsq - self.count * true_mean ** 2

        resid = self.pred_sumsq - 2 * self.product + self.true_sumsq
        r2 = 1 - resid / total

        if self.summarize:
            return r2.mean()
        else:
            return r2

    def reset(self):
        self.count.zero_()
        self.true_sum.zero_()
        self.true_sumsq.zero_()
        self.product.zero_()
        self.pred_sumsq.zero_()
