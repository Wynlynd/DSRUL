import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from ccar import ClusterContrastiveRelabeler, ccar_loss
from objectives import (
    auem_regularization_loss,
    ccar_batch_targets,
    dsruc_loss,
    split_uncertainty_regions,
)
from resnet import resnet18


class AUEMTests(unittest.TestCase):
    @staticmethod
    def build_model(out_dimension=32):
        args = SimpleNamespace(out_dimension=out_dimension)
        return resnet18(
            pretrained=False, drop_rate=0.0, args=args,
            out_dim=out_dimension)

    @staticmethod
    def bypass_trunk(model):
        for name in ('conv1', 'bn1', 'relu', 'maxpool',
                     'layer1', 'layer2', 'layer3', 'layer4'):
            setattr(model, name, nn.Identity())

    def test_importance_head_consumes_normalized_mu(self):
        model = self.build_model()
        self.assertEqual(model.alpha[0].in_features, 32)

        # Bypass the convolutional trunk so this test locks the AUEM dataflow
        # without paying for a full ResNet forward pass.
        self.bypass_trunk(model)
        model.eval()

        captured = []
        handle = model.alpha[0].register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone()))
        try:
            features = torch.randn(2, 512, 7, 7)
            with torch.no_grad():
                mu, uncertainty, importance = model(
                    features, return_importance=True)
        finally:
            handle.remove()

        expected_min = mu.amin(dim=1, keepdim=True)
        expected_range = (mu.amax(dim=1, keepdim=True)
                          - expected_min).clamp_min(1e-6)
        expected_normalized_mu = (mu - expected_min) / expected_range

        self.assertEqual(mu.shape, (2, 32))
        self.assertEqual(uncertainty.shape, (2, 32))
        self.assertEqual(importance.shape, (2, 1))
        torch.testing.assert_close(captured[0], expected_normalized_mu)
        self.assertTrue(torch.isfinite(importance).all().item())
        self.assertTrue(((importance >= 0) & (importance <= 1)).all().item())

    def test_importance_path_is_trainable_and_constant_safe(self):
        model = self.build_model(out_dimension=4)
        mu = torch.tensor([[0.0, 1.0, 2.0, 3.0],
                           [5.0, 5.0, 5.0, 5.0]], requires_grad=True)

        importance = model.compute_importance(mu)
        self.assertTrue(torch.isfinite(importance).all().item())
        importance.sum().backward()

        self.assertIsNotNone(mu.grad)
        self.assertTrue(torch.isfinite(mu.grad).all().item())
        self.assertIsNotNone(model.alpha[0].weight.grad)
        self.assertTrue(torch.isfinite(model.alpha[0].weight.grad).all().item())
        self.assertGreater(model.alpha[0].weight.grad.abs().sum().item(), 0.0)

    def test_three_term_objective_reaches_all_learned_branches(self):
        devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        for device in devices:
            with self.subTest(device=device):
                model = self.build_model(out_dimension=8).to(device)
                self.bypass_trunk(model)
                model.train()
                classifier = nn.Linear(8, 7).to(device)
                ccar = ClusterContrastiveRelabeler(7, 8).to(device)

                features = torch.randn(
                    4, 512, 7, 7, device=device, requires_grad=True)
                labels = torch.tensor([0, 1, 2, 3], device=device)
                mu, uncertainty, importance = model(
                    features, return_importance=True)
                importance = importance.squeeze(1)
                high_uncertainty, low_uncertainty = split_uncertainty_regions(
                    importance)

                auem = auem_regularization_loss(
                    importance, high_uncertainty, low_uncertainty)
                dsruc, pair_count = dsruc_loss(
                    mu, uncertainty, labels, classifier,
                    high_uncertainty, low_uncertainty)
                logits = classifier(mu)
                targets = ccar_batch_targets(
                    ccar, mu, labels, importance, logits,
                    relabel_enabled=False)
                loss = ccar_loss(logits, targets, importance) + auem + dsruc
                loss.backward()

                self.assertGreater(pair_count, 0)
                checked_gradients = (
                    features.grad,
                    model.mu[4].weight.grad,
                    model.log_var[4].weight.grad,
                    model.alpha[0].weight.grad,
                    classifier.weight.grad,
                )
                for gradient in checked_gradients:
                    self.assertIsNotNone(gradient)
                    self.assertTrue(torch.isfinite(gradient).all().item())
                    self.assertGreater(gradient.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
