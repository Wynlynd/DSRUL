import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccar import ClusterContrastiveRelabeler
from objectives import (
    auem_regularization_loss,
    build_dsruc_pairs,
    ccar_batch_targets,
    dsruc_loss,
    mix_embeddings_by_uncertainty,
    split_uncertainty_regions,
)


class ObjectiveTests(unittest.TestCase):
    def test_auem_split_and_loss_match_equations_2_and_3(self):
        importance = torch.tensor([0.8, 0.1, 0.6, 0.2, 0.9])
        high_uncertainty, low_uncertainty = split_uncertainty_regions(importance)

        self.assertEqual(set(high_uncertainty.tolist()), {1, 3})
        self.assertEqual(set(low_uncertainty.tolist()), {0, 2, 4})
        actual = auem_regularization_loss(
            importance, high_uncertainty, low_uncertainty, delta=0.5)
        expected = torch.log(
            (torch.tensor(0.15) + 0.5) / torch.tensor((0.8 + 0.6 + 0.9) / 3))
        torch.testing.assert_close(actual, expected)

    def test_dsruc_mix_and_loss_match_equations_4_to_6(self):
        embeddings = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        uncertainty = torch.tensor(
            [[0.2, 0.8], [0.2, 0.2]], requires_grad=True)
        labels = torch.tensor([0, 1])
        classifier = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))

        mixed, high_weight, low_weight = mix_embeddings_by_uncertainty(
            embeddings[[0]], embeddings[[1]],
            uncertainty[[0]], uncertainty[[1]])
        torch.testing.assert_close(high_weight, torch.tensor([0.65]))
        torch.testing.assert_close(low_weight, torch.tensor([0.35]))
        torch.testing.assert_close(mixed, torch.tensor([[1.3, 0.7]]))

        actual, pair_count = dsruc_loss(
            embeddings, uncertainty, labels, classifier,
            torch.tensor([0]), torch.tensor([1]))
        expected_logits = classifier(torch.tensor([[1.3, 0.7]]))
        expected = (F.cross_entropy(expected_logits, torch.tensor([0]))
                    + F.cross_entropy(expected_logits, torch.tensor([1])))
        self.assertEqual(pair_count, 1)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(uncertainty.grad.abs().sum().item(), 0.0)

    def test_dsruc_rejects_same_label_pairs(self):
        labels = torch.tensor([0, 0])
        high, low = build_dsruc_pairs(
            torch.tensor([0]), torch.tensor([1]), labels)
        self.assertEqual(high.numel(), 0)
        self.assertEqual(low.numel(), 0)

        embeddings = torch.randn(2, 3, requires_grad=True)
        classifier = nn.Linear(3, 2)
        loss, pair_count = dsruc_loss(
            embeddings, torch.tensor([[0.7, 0.3], [0.3, 0.7]]), labels, classifier,
            torch.tensor([0]), torch.tensor([1]))
        self.assertEqual(pair_count, 0)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_ccar_proposes_with_original_labels_before_update(self):
        events = []

        class RecordingCCAR:
            def update(self, _embeddings, labels):
                events.append(('update', labels.clone()))

            def propose(self, _embeddings, labels, _importance, _logits):
                events.append(('propose', labels.clone()))
                return torch.tensor([True, False]), torch.tensor([1, 1])

        original = torch.tensor([0, 1])
        corrected = ccar_batch_targets(
            RecordingCCAR(), torch.randn(2, 3), original,
            torch.tensor([0.2, 0.8]), torch.randn(2, 2), True)

        self.assertEqual([event[0] for event in events], ['propose', 'update'])
        torch.testing.assert_close(events[0][1], original)
        torch.testing.assert_close(events[1][1], original)
        torch.testing.assert_close(corrected, torch.tensor([1, 1]))
        torch.testing.assert_close(original, torch.tensor([0, 1]))

    def test_ccar_only_updates_before_relabeling_is_enabled(self):
        events = []

        class RecordingCCAR:
            def update(self, _embeddings, labels):
                events.append(('update', labels.clone()))

            def propose(self, _embeddings, labels, _importance, _logits):
                events.append(('propose', labels.clone()))
                return torch.tensor([True]), torch.tensor([1])

        original = torch.tensor([0])
        targets = ccar_batch_targets(
            RecordingCCAR(), torch.randn(1, 3), original,
            torch.tensor([0.5]), torch.randn(1, 2), False)

        self.assertEqual([event[0] for event in events], ['update'])
        torch.testing.assert_close(events[0][1], original)
        self.assertIs(targets, original)

    def test_baseline_updates_before_proposal(self):
        events = []

        class RecordingCCAR:
            def update(self, _embeddings, _labels):
                events.append('update')

            def propose(self, _embeddings, _labels, _importance, _logits):
                events.append('propose')
                return torch.tensor([False]), torch.tensor([0])

        ccar_batch_targets(
            RecordingCCAR(), torch.randn(1, 3), torch.tensor([0]),
            torch.tensor([0.5]), torch.randn(1, 2), True,
            variant='baseline')

        self.assertEqual(events, ['update', 'propose'])

    def test_oracle_soft_forwards_weights_without_changing_observed_target(self):
        events = []

        class RecordingCCAR:
            def update(self, _embeddings, labels, weights=None):
                events.append(('update', labels.clone(), weights.clone()))

            def propose(self, _embeddings, labels, _importance, _logits):
                events.append(('propose', labels.clone()))
                return torch.tensor([False, False]), torch.tensor([1, 0])

        original = torch.tensor([0, 1])
        weights = torch.tensor([1.0, 0.5])
        targets, details = ccar_batch_targets(
            RecordingCCAR(), torch.randn(2, 3), original,
            torch.tensor([0.5, 0.5]), torch.randn(2, 2), True,
            variant='oracle_soft', prototype_weights=weights,
            return_diagnostics=True)

        self.assertEqual([event[0] for event in events], ['propose', 'update'])
        torch.testing.assert_close(events[1][1], original)
        torch.testing.assert_close(events[1][2], weights)
        self.assertIs(targets, original)
        self.assertFalse(details['relabel_mask'].any().item())

    def test_all_variants_only_update_before_relabel_epoch(self):
        for variant in ('baseline', 'preupdate', 'oracle_soft'):
            events = []

            class RecordingCCAR:
                def update(self, _embeddings, _labels):
                    events.append('update')

                def propose(self, *_args):
                    events.append('propose')
                    return torch.tensor([True]), torch.tensor([1])

            targets = ccar_batch_targets(
                RecordingCCAR(), torch.randn(1, 3), torch.tensor([0]),
                torch.tensor([0.5]), torch.randn(1, 2), False,
                variant=variant)
            self.assertEqual(events, ['update'])
            torch.testing.assert_close(targets, torch.tensor([0]))

    def test_ccar_decision_uses_the_preupdate_prototypes(self):
        ccar = ClusterContrastiveRelabeler(
            num_classes=2, feature_dim=1, momentum=0.0,
            temperature=0.1, beta=0.2, gamma=0.0, alpha=0.4)
        ccar.update(
            torch.tensor([[0.0], [10.0]]), torch.tensor([0, 1]))

        targets = ccar_batch_targets(
            ccar,
            torch.tensor([[9.0]]),
            torch.tensor([0]),
            torch.tensor([0.5]),
            torch.tensor([[0.0, 0.30228087]]),
            relabel_enabled=True)

        torch.testing.assert_close(targets, torch.tensor([1]))
        torch.testing.assert_close(ccar.prototypes[0], torch.tensor([9.0]))


if __name__ == '__main__':
    unittest.main()
