import torch
import torch.nn as nn
import torch.nn.functional as F


def ccar_loss(logits, labels, importance):
    """Equation (13): scale logits by importance and train with the supplied target.

    For a relabeled sample, ``labels`` should contain its current-batch kappa-prime;
    the dataset's original labels remain available to the geometry and P_gt steps.
    """
    return F.cross_entropy(logits * importance.reshape(-1, 1), labels)


class ClusterContrastiveRelabeler(nn.Module):
    def __init__(self, num_classes, feature_dim, momentum=0.9, temperature=0.1,
                 beta=0.2, gamma=0.6, alpha=0.4):
        super().__init__()
        if not 0 <= momentum < 1 or temperature <= 0 or beta < 0:
            raise ValueError('Invalid CCAR momentum, temperature, or beta.')
        if not 0 <= gamma < 1 or not 0 <= alpha <= 1:
            raise ValueError('CCAR requires 0 <= gamma < 1 and 0 <= alpha <= 1.')
        self.momentum = momentum
        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim))
        self.register_buffer('initialized', torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, features, labels):
        """Equation (7): update only classes present in the current batch."""
        features = features.detach()
        for class_id in labels.unique().tolist():
            batch_center = features[labels == class_id].mean(dim=0)
            if self.initialized[class_id]:
                self.prototypes[class_id].mul_(self.momentum).add_(
                    batch_center, alpha=1 - self.momentum)
            else:
                # The paper leaves P_c^(0) unspecified; the first batch mean is unbiased.
                self.prototypes[class_id].copy_(batch_center)
                self.initialized[class_id] = True

    @torch.no_grad()
    def thresholds(self, features, labels, importance):
        """Equations (8)-(11), with original labels and importance kept distinct."""
        distances = torch.cdist(features.detach(), self.prototypes)
        distances[:, ~self.initialized] = float('inf')
        valid = self.initialized[labels] & (self.initialized.sum() >= 2)

        intra = distances.gather(1, labels[:, None]).squeeze(1)
        intra = torch.where(valid, intra, torch.zeros_like(intra))
        inter_distances = distances.clone()
        inter_distances.scatter_(1, labels[:, None], float('inf'))
        inter = inter_distances.min(dim=1).values
        inter = torch.where(valid, inter, torch.zeros_like(inter))

        phi = torch.sigmoid((intra - inter) / self.temperature)
        batch_mean = importance.mean()
        denominator = 1 + self.gamma * torch.sign(importance - batch_mean)
        threshold = self.beta / denominator * (1 - self.alpha * phi)
        return threshold, valid

    @torch.no_grad()
    def propose(self, features, labels, importance, logits):
        """Equation (12): propose a batch-local label from classifier probabilities."""
        threshold, valid = self.thresholds(features, labels, importance)
        probabilities = torch.softmax(logits.detach(), dim=1)
        max_probability, predicted = probabilities.max(dim=1)
        given_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
        relabel = valid & (max_probability > given_probability + threshold)
        return relabel, predicted
