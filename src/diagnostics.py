import numpy as np
import torch


def classification_metrics(targets, predictions, num_classes):
    """Compute deterministic classification metrics from integer labels."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError('targets and predictions must have matching shapes')
    if targets.size == 0:
        raise ValueError('classification metrics require at least one sample')
    if ((targets < 0).any() or (targets >= num_classes).any()
            or (predictions < 0).any()
            or (predictions >= num_classes).any()):
        raise ValueError('labels must be in [0, num_classes)')

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)

    recall = np.divide(
        true_positive, support, out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0)
    precision = np.divide(
        true_positive, predicted_count,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_count > 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0)
    supported = support > 0

    return {
        'accuracy': float(true_positive.sum() / targets.size),
        'macro_f1': float(f1[supported].mean()),
        'balanced_accuracy': float(recall[supported].mean()),
        'per_class_recall': recall.tolist(),
        'confusion_matrix': confusion.tolist(),
    }


def relabel_diagnostic_tensors(original_labels, clean_labels, relabel_mask,
                               predicted_labels, num_classes):
    """Return device-resident counters for one batch."""
    tensors = [
        original_labels.detach(), clean_labels.detach(), relabel_mask.detach(),
        predicted_labels.detach(),
    ]
    original, clean, relabel, predicted = [tensor.reshape(-1) for tensor in tensors]
    if not all(tensor.shape == original.shape for tensor in (clean, relabel, predicted)):
        raise ValueError('relabel diagnostic tensors must have matching shapes')
    if relabel.dtype != torch.bool:
        raise ValueError('relabel_mask must be boolean')

    noisy = original != clean
    corrected = relabel & noisy & (predicted == clean)
    false_correction = relabel & ~noisy & (predicted != clean)
    wrong_to_wrong = relabel & noisy & (predicted != clean)
    still_wrong = relabel & (predicted != clean)

    def by_observed_class(mask):
        return torch.bincount(original[mask], minlength=num_classes)

    return {
        'sample_count': torch.tensor(
            original.numel(), device=original.device, dtype=torch.long),
        'noisy_sample_count': noisy.sum(),
        'relabel_count': relabel.sum(),
        'corrected_noisy_count': corrected.sum(),
        'false_correction_count': false_correction.sum(),
        'wrong_to_wrong_count': wrong_to_wrong.sum(),
        'relabel_still_wrong_count': still_wrong.sum(),
        'relabel_by_observed_class': by_observed_class(relabel),
        'corrected_by_observed_class': by_observed_class(corrected),
        'false_correction_by_observed_class': by_observed_class(false_correction),
        'still_wrong_by_observed_class': by_observed_class(still_wrong),
    }


def relabel_diagnostics_to_python(metrics):
    """Transfer one accumulated diagnostic record to JSON-safe values."""
    return {
        key: (value.detach().cpu().tolist() if value.ndim else int(value.item()))
        for key, value in metrics.items()
    }


def relabel_diagnostics(original_labels, clean_labels, relabel_mask,
                        predicted_labels, num_classes):
    """Measure relabel quality without feeding clean labels into the loss."""
    return relabel_diagnostics_to_python(relabel_diagnostic_tensors(
        original_labels, clean_labels, relabel_mask, predicted_labels,
        num_classes))


def merge_relabel_diagnostics(total, batch):
    """Add one batch's relabel counters to an epoch accumulator."""
    if total is None:
        return {
            key: value.clone()
            for key, value in batch.items()
        }
    for key, value in batch.items():
        total[key] += value
    return total
