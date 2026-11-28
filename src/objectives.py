import torch
import torch.nn.functional as F


def split_uncertainty_regions(importance):
    """Split a batch into the two 50% regions defined after Equation (1)."""
    if importance.ndim != 1:
        raise ValueError('importance must have shape [batch]')
    if importance.numel() < 2:
        raise ValueError('AUEM requires at least two samples per batch')

    high_uncertainty_count = importance.numel() // 2
    sorted_indices = torch.argsort(importance)
    high_uncertainty_indices = sorted_indices[:high_uncertainty_count]
    low_uncertainty_indices = sorted_indices[high_uncertainty_count:].flip(0)
    return high_uncertainty_indices, low_uncertainty_indices


def auem_regularization_loss(importance, high_uncertainty_indices,
                             low_uncertainty_indices, delta=0.5):
    """Equations (2)-(3), using group means implied by the surrounding text."""
    low_importance_mean = importance[high_uncertainty_indices].mean()
    high_importance_mean = importance[low_uncertainty_indices].mean()
    return torch.log(
        (low_importance_mean + delta) / high_importance_mean.clamp_min(1e-6))


def build_dsruc_pairs(high_uncertainty_indices, low_uncertainty_indices,
                      labels, generator=None):
    """Randomly pair the two regions and retain different-label pairs only."""
    pair_count = min(high_uncertainty_indices.numel(),
                     low_uncertainty_indices.numel())
    if pair_count == 0:
        return high_uncertainty_indices[:0], low_uncertainty_indices[:0]

    high_order = torch.randperm(
        high_uncertainty_indices.numel(), device=high_uncertainty_indices.device,
        generator=generator)[:pair_count]
    low_order = torch.randperm(
        low_uncertainty_indices.numel(), device=low_uncertainty_indices.device,
        generator=generator)[:pair_count]
    high_pair_indices = high_uncertainty_indices[high_order]
    low_pair_indices = low_uncertainty_indices[low_order]

    different_labels = labels[high_pair_indices] != labels[low_pair_indices]
    return (high_pair_indices[different_labels],
            low_pair_indices[different_labels])


def mix_embeddings_by_uncertainty(high_embeddings, low_embeddings,
                                  high_uncertainty, low_uncertainty):
    """Equations (4)-(5): normalize uncertainty vectors, then mix by their means."""
    if high_embeddings.ndim != 2 or low_embeddings.ndim != 2:
        raise ValueError('embeddings must have shape [pairs, features]')
    if high_embeddings.shape != low_embeddings.shape:
        raise ValueError('high and low embeddings must have matching shapes')
    if high_uncertainty.shape != high_embeddings.shape:
        raise ValueError(
            'high uncertainty must match the high embedding shape')
    if low_uncertainty.shape != low_embeddings.shape:
        raise ValueError(
            'low uncertainty must match the low embedding shape')

    denominator = (high_uncertainty + low_uncertainty).clamp_min(1e-8)
    normalized_high = high_uncertainty / denominator
    normalized_low = low_uncertainty / denominator
    high_weight = normalized_high.mean(dim=1)
    low_weight = normalized_low.mean(dim=1)
    mixed = (high_weight.unsqueeze(1) * high_embeddings
             + low_weight.unsqueeze(1) * low_embeddings)
    return mixed, high_weight, low_weight


def dsruc_loss(embeddings, uncertainty, labels, classifier,
               high_uncertainty_indices, low_uncertainty_indices,
               generator=None):
    """Equations (4)-(6), preserving the sum of the two mean CE terms."""
    high_pairs, low_pairs = build_dsruc_pairs(
        high_uncertainty_indices, low_uncertainty_indices, labels, generator)
    if high_pairs.numel() == 0:
        return embeddings.sum() * 0.0, 0

    mixed, _, _ = mix_embeddings_by_uncertainty(
        embeddings[high_pairs], embeddings[low_pairs],
        uncertainty[high_pairs], uncertainty[low_pairs])
    mixed_logits = classifier(mixed)
    loss = (F.cross_entropy(mixed_logits, labels[high_pairs])
            + F.cross_entropy(mixed_logits, labels[low_pairs]))
    return loss, high_pairs.numel()


def ccar_batch_targets(ccar, embeddings, original_labels, importance, logits,
                       relabel_enabled, variant='preupdate',
                       prototype_weights=None, return_diagnostics=False):
    """Build batch-local CCAR targets for the selected prototype variant."""
    if variant not in ('baseline', 'preupdate', 'oracle_soft'):
        raise ValueError('Unknown CCAR variant: {}'.format(variant))

    def update_prototypes():
        if prototype_weights is None:
            ccar.update(embeddings, original_labels)
        else:
            ccar.update(embeddings, original_labels, weights=prototype_weights)

    if not relabel_enabled:
        update_prototypes()
        relabel_mask = torch.zeros_like(original_labels, dtype=torch.bool)
        predicted_labels = original_labels.clone()
        corrected_labels = original_labels
    else:
        if variant == 'baseline':
            update_prototypes()
            relabel_mask, predicted_labels = ccar.propose(
                embeddings, original_labels, importance, logits)
        else:
            relabel_mask, predicted_labels = ccar.propose(
                embeddings, original_labels, importance, logits)
            update_prototypes()

        if relabel_mask.any():
            corrected_labels = original_labels.clone()
            corrected_labels[relabel_mask] = predicted_labels[relabel_mask]
        else:
            corrected_labels = original_labels

    if not return_diagnostics:
        return corrected_labels
    return corrected_labels, {
        'relabel_mask': relabel_mask.detach(),
        'predicted_labels': predicted_labels.detach(),
    }
