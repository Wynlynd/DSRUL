import unittest
import math

import torch
import torch.nn.functional as F

from ccar import ClusterContrastiveRelabeler, ccar_loss


class CCARTests(unittest.TestCase):
    def test_prototype_ema_and_missing_classes(self):
        ccar = ClusterContrastiveRelabeler(3, 2, momentum=0.9)
        ccar.update(torch.tensor([[0., 0.], [2., 0.], [4., 0.]]),
                    torch.tensor([0, 0, 1]))
        torch.testing.assert_close(ccar.prototypes[0], torch.tensor([1., 0.]))
        torch.testing.assert_close(ccar.prototypes[1], torch.tensor([4., 0.]))
        self.assertFalse(ccar.initialized[2].item())

        ccar.update(torch.tensor([[3., 0.]]), torch.tensor([0]))
        torch.testing.assert_close(ccar.prototypes[0], torch.tensor([1.2, 0.]))
        torch.testing.assert_close(ccar.prototypes[1], torch.tensor([4., 0.]))
        restored = ClusterContrastiveRelabeler(3, 2)
        restored.load_state_dict(ccar.state_dict())
        torch.testing.assert_close(restored.prototypes, ccar.prototypes)
        torch.testing.assert_close(restored.initialized, ccar.initialized)

    def test_weighted_prototype_center_matches_manual_calculation(self):
        ccar = ClusterContrastiveRelabeler(2, 2, momentum=0.0)
        features = torch.tensor([[0., 2.], [4., 0.], [10., 10.]])
        labels = torch.tensor([0, 0, 1])
        weights = torch.tensor([1.0, 0.5, 0.0])

        ccar.update(features, labels, weights=weights)

        torch.testing.assert_close(
            ccar.prototypes[0], torch.tensor([4. / 3., 4. / 3.]))
        self.assertFalse(ccar.initialized[1].item())

        unweighted = ClusterContrastiveRelabeler(2, 2, momentum=0.0)
        all_ones = ClusterContrastiveRelabeler(2, 2, momentum=0.0)
        unweighted.update(features, labels)
        all_ones.update(features, labels, weights=torch.ones(3))
        torch.testing.assert_close(
            unweighted.prototypes, all_ones.prototypes)
        torch.testing.assert_close(
            unweighted.initialized, all_ones.initialized)

    def test_weighted_update_validates_weights(self):
        ccar = ClusterContrastiveRelabeler(2, 2)
        features = torch.zeros(2, 2)
        labels = torch.tensor([0, 1])
        with self.assertRaisesRegex(ValueError, 'shape'):
            ccar.update(features, labels, weights=torch.ones(2, 1))
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            ccar.update(features, labels, weights=torch.tensor([1., -1.]))
        with self.assertRaisesRegex(ValueError, 'finite'):
            ccar.update(features, labels, weights=torch.tensor([1., float('nan')]))

    def test_vectorized_weighted_updates_match_reference(self):
        ccar = ClusterContrastiveRelabeler(4, 2, momentum=0.75)
        ccar.update(
            torch.tensor([[2., 0.], [0., 4.], [8., 8.]]),
            torch.tensor([0, 1, 2]))
        previous = ccar.prototypes.clone()

        features = torch.tensor([
            [4., 2.], [10., 6.], [2., 8.], [100., 100.]])
        labels = torch.tensor([0, 0, 2, 3])
        weights = torch.tensor([1.0, 0.5, 0.25, 0.0])
        ccar.update(features, labels, weights=weights)

        center_0 = (features[0] + 0.5 * features[1]) / 1.5
        center_2 = features[2]
        expected = previous.clone()
        expected[0] = 0.75 * previous[0] + 0.25 * center_0
        expected[2] = 0.75 * previous[2] + 0.25 * center_2
        torch.testing.assert_close(ccar.prototypes, expected)
        self.assertEqual(ccar.initialized.tolist(), [True, True, True, False])

    def test_geometry_and_importance_control_threshold(self):
        for device in self.devices():
            with self.subTest(device=device):
                ccar = ClusterContrastiveRelabeler(2, 2).to(device)
                ccar.update(torch.tensor([[0., 0.], [2., 0.]], device=device),
                            torch.tensor([0, 1], device=device))
                features = torch.tensor([[1.5, 0.], [1.5, 0.]], device=device)
                labels = torch.tensor([0, 0], device=device)
                importance = torch.tensor([0.8, 0.2], device=device)
                thresholds, valid = ccar.thresholds(features, labels, importance)
                phi = torch.sigmoid(torch.tensor(10., device=device))
                expected = torch.tensor([0.2 / 1.6, 0.2 / 0.4], device=device) * (
                    1 - 0.4 * phi)
                torch.testing.assert_close(thresholds, expected)
                self.assertTrue(valid.all().item())

                logits = torch.tensor([[0., math.log(1.5)],
                                       [0., math.log(1.5)]], device=device)
                relabel, predictions = ccar.propose(features, labels, importance, logits)
                self.assertEqual(relabel.tolist(), [True, False])
                self.assertEqual(predictions.tolist(), [1, 1])

                equal_importance = torch.tensor([0.5, 0.5], device=device)
                easy_and_inconsistent = torch.tensor([[0.1, 0.], [1.5, 0.]], device=device)
                geometric_thresholds, _ = ccar.thresholds(
                    easy_and_inconsistent, labels, equal_importance)
                self.assertLess(geometric_thresholds[1].item(),
                                geometric_thresholds[0].item())

    def test_uninitialized_prototypes_cannot_trigger_relabel(self):
        ccar = ClusterContrastiveRelabeler(2, 2)
        ccar.update(torch.tensor([[0., 0.]]), torch.tensor([0]))
        mask, _ = ccar.propose(torch.tensor([[0., 0.]]), torch.tensor([0]),
                               torch.tensor([0.5]), torch.tensor([[0., 10.]]))
        self.assertFalse(mask.any().item())

    def test_scaled_logit_loss_and_gradients(self):
        for device in self.devices():
            with self.subTest(device=device):
                logits = torch.tensor([[2., 0.], [0., 2.]], device=device,
                                      requires_grad=True)
                importance = torch.tensor([0.8, 0.3], device=device,
                                          requires_grad=True)
                labels = torch.tensor([0, 1], device=device)
                loss = ccar_loss(logits, labels, importance)
                scaled_logits = logits * importance[:, None]
                expected = (torch.logsumexp(scaled_logits, dim=1)
                            - scaled_logits.gather(1, labels[:, None]).squeeze(1)).mean()
                torch.testing.assert_close(loss, expected)
                old_weighted_ce = (importance * F.cross_entropy(
                    logits, labels, reduction='none')).sum() / importance.sum()
                self.assertGreater(abs((loss - old_weighted_ce).item()), 1e-3)
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all().item())
                self.assertTrue(torch.isfinite(importance.grad).all().item())
                self.assertTrue((importance.grad != 0).all().item())

    @staticmethod
    def devices():
        return ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']


if __name__ == '__main__':
    unittest.main()
