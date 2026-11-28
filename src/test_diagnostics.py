import unittest

import torch

from diagnostics import classification_metrics, relabel_diagnostics


class DiagnosticTests(unittest.TestCase):
    def test_classification_metrics_match_hand_calculation(self):
        metrics = classification_metrics(
            [0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 0, 2], num_classes=3)

        self.assertEqual(
            metrics['confusion_matrix'], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(metrics['accuracy'], 4 / 6)
        self.assertAlmostEqual(metrics['balanced_accuracy'], (0.5 + 1 + 0.5) / 3)
        expected_f1 = (0.5 + 0.8 + 2 / 3) / 3
        self.assertAlmostEqual(metrics['macro_f1'], expected_f1)

    def test_relabel_diagnostics_separate_corrections_and_new_errors(self):
        original = torch.tensor([0, 1, 2, 0, 1])
        clean = torch.tensor([0, 2, 2, 1, 0])
        relabel = torch.tensor([True, True, False, True, True])
        predicted = torch.tensor([1, 2, 2, 2, 2])

        metrics = relabel_diagnostics(
            original, clean, relabel, predicted, num_classes=3)

        self.assertEqual(metrics['noisy_sample_count'], 3)
        self.assertEqual(metrics['relabel_count'], 4)
        self.assertEqual(metrics['corrected_noisy_count'], 1)
        self.assertEqual(metrics['false_correction_count'], 1)
        self.assertEqual(metrics['wrong_to_wrong_count'], 2)
        self.assertEqual(metrics['relabel_still_wrong_count'], 3)
        self.assertEqual(metrics['relabel_by_observed_class'], [2, 2, 0])


if __name__ == '__main__':
    unittest.main()
