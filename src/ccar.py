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
    def update(self, features, labels, weights=None):
        """Update present-class prototypes, optionally using sample weights."""
        features = features.detach()
        if weights is not None:
            if weights.ndim != 1 or weights.shape[0] != features.shape[0]:
                raise ValueError('weights must have shape [batch]')
            weights = weights.detach().to(
                device=features.device, dtype=features.dtype)
            valid_weights = torch.isfinite(weights).all() & (weights >= 0).all()
            if weights.is_cuda:
                torch._assert_async(
                    valid_weights, 'weights must be finite and non-negative')
            elif not valid_weights:
                raise ValueError('weights must be finite and non-negative')
        else:
            weights = torch.ones(
                features.shape[0], device=features.device,
                dtype=features.dtype)

        num_classes = self.prototypes.shape[0]
        weight_sums = torch.zeros(
            num_classes, device=features.device, dtype=features.dtype)
        weighted_feature_sums = torch.zeros(
            num_classes, features.shape[1], device=features.device,
            dtype=features.dtype)
        weight_sums.index_add_(0, labels, weights)
        weighted_feature_sums.index_add_(
            0, labels, features * weights.unsqueeze(1))

        present = weight_sums > 0
        denominators = torch.where(
            present, weight_sums, torch.ones_like(weight_sums))
        batch_centers = weighted_feature_sums / denominators.unsqueeze(1)
        updated = torch.where(
            self.initialized.unsqueeze(1),
            self.prototypes * self.momentum
            + batch_centers * (1 - self.momentum),
            batch_centers)
        self.prototypes.copy_(torch.where(
            present.unsqueeze(1), updated, self.prototypes))
        self.initialized.logical_or_(present)

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
