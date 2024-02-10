import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SCAdaCos(nn.Module):
    def __init__(self, n_classes=10, n_subclusters=1, input_shape=256, trainable=False):
        super(SCAdaCos, self).__init__()
        self.n_classes = n_classes
        self.n_subclusters = n_subclusters
        self.s_init = torch.sqrt(torch.tensor(2.0)) * torch.log(
            torch.tensor(n_classes * n_subclusters - 1.0)
        )
        self.trainable = trainable

        self.W = nn.Parameter(
            torch.randn((input_shape, self.n_classes * self.n_subclusters))
        )
        nn.init.xavier_uniform_(self.W)
        self.W.requires_grad = self.trainable

        self.s = nn.Parameter(self.s_init, requires_grad=False)

    def forward(self, inputs):
        x, y1, y2 = inputs
        y1_orig = y1.clone()
        # y1 = y1.repeat(self.n_subclusters, dim=-1)
        y1 = y1.repeat(*([1] * (y1.dim() - 1) + [self.n_subclusters]))
        # normalize feature
        x = F.normalize(x, p=2, dim=1)
        # normalize weights
        W = F.normalize(self.W, p=2, dim=0)
        # dot product
        logits = x @ W  # same as cos theta
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))

        if self.training:
            max_s_logits = torch.max(self.s * logits)
            B_avg = torch.exp(self.s * logits - max_s_logits)
            B_avg = torch.mean(torch.sum(B_avg, dim=1))
            theta_class = (
                torch.sum(y1 * theta, dim=1)
                * torch.count_nonzero(y1_orig, dim=1).float()
            )  # take mix-upped angle of mix-upped classes
            theta_med = torch.quantile(theta_class, q=0.5)  # computes median
            self.s.data = (max_s_logits + torch.log(B_avg)) / torch.cos(
                torch.min(torch.tensor(math.pi / 4), theta_med)
            ) + 1e-7
        logits *= self.s
        out = F.softmax(logits, dim=1)
        out = out.view(-1, self.n_classes, self.n_subclusters)
        out = torch.sum(out, dim=2)
        loss = F.cross_entropy(
            input=out, target=y1_orig.argmax(dim=1), reduction="none"
        )
        return loss
