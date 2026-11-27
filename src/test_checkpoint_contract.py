import unittest

import torch
import torch.nn as nn

from train_rafdb import (
    CHECKPOINT_FORMAT_VERSION,
    MinimumSizeBatchSampler,
    PROTOTYPE_UPDATE_POLICY,
    checkpoint_state_dict,
    load_shape_compatible_state,
    method_contract,
    parse_args,
    restore_ema_states,
    validate_resume_checkpoint,
)


class CheckpointContractTests(unittest.TestCase):
    @staticmethod
    def args(**overrides):
        args = parse_args(['--raf_path', 'raf-basic'])
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    @staticmethod
    def checkpoint(args):
        checkpoint = {
            'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
            'implementation_version': '2.0.0-dev',
            'method_contract': method_contract(args),
            'config': vars(args).copy(),
            'iter': 3,
            'best_val_accuracy': 0.5,
            'model_state_dict': {'weight': torch.tensor([1.0])},
            'fc_state_dict': {'weight': torch.tensor([2.0])},
            'ccar_state_dict': {},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'ema_decay': args.ema_decay,
        }
        if args.ema_decay > 0.0:
            checkpoint['resume_model_state_dict'] = {'weight': torch.tensor([3.0])}
            checkpoint['resume_fc_state_dict'] = {'weight': torch.tensor([4.0])}
        return checkpoint

    def test_v2_resume_requires_matching_method_contract(self):
        args = self.args()
        checkpoint = self.checkpoint(args)
        validate_resume_checkpoint(checkpoint, args)
        self.assertEqual(
            checkpoint['method_contract']['prototype_update_policy'],
            PROTOTYPE_UPDATE_POLICY)

        with self.assertRaisesRegex(ValueError, 'not a V2 checkpoint'):
            validate_resume_checkpoint({}, args)

        mismatched = dict(checkpoint)
        mismatched['method_contract'] = dict(checkpoint['method_contract'])
        mismatched['method_contract']['embedding_dim'] = 32
        with self.assertRaisesRegex(ValueError, 'method contract mismatch'):
            validate_resume_checkpoint(mismatched, args)

        mismatched = dict(checkpoint)
        mismatched['method_contract'] = dict(checkpoint['method_contract'])
        mismatched['method_contract']['prototype_update_policy'] = 'all_samples'
        with self.assertRaisesRegex(ValueError, 'method contract mismatch'):
            validate_resume_checkpoint(mismatched, args)

        mismatched = dict(checkpoint)
        mismatched['method_contract'] = dict(checkpoint['method_contract'])
        mismatched['method_contract']['uncertainty_representation'] = 'scalar'
        with self.assertRaisesRegex(ValueError, 'method contract mismatch'):
            validate_resume_checkpoint(mismatched, args)

    def test_resume_rejects_config_drift_and_incomplete_state(self):
        args = self.args()
        checkpoint = self.checkpoint(args)

        changed_args = self.args(beta=0.3)
        with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
            validate_resume_checkpoint(checkpoint, changed_args)

        relocated_args = self.args(
            checkpoint='old-run.pth', save_dir='new-output',
            skip_test_eval=True, target_val_accuracy=0.99)
        validate_resume_checkpoint(checkpoint, relocated_args)

        extended_args = self.args(epochs=args.epochs + 10)
        with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
            validate_resume_checkpoint(checkpoint, extended_args)

        incomplete = dict(checkpoint)
        incomplete.pop('scheduler_state_dict')
        with self.assertRaisesRegex(ValueError, 'scheduler_state_dict'):
            validate_resume_checkpoint(incomplete, args)

        ema_args = self.args(ema_decay=0.9)
        incomplete_ema = self.checkpoint(ema_args)
        incomplete_ema.pop('resume_model_state_dict')
        with self.assertRaisesRegex(ValueError, 'resume_model_state_dict'):
            validate_resume_checkpoint(incomplete_ema, ema_args)

    def test_ema_history_is_restored_as_an_independent_copy(self):
        args = self.args(ema_decay=0.9)
        checkpoint = self.checkpoint(args)
        model_ema, fc_ema = restore_ema_states(checkpoint, args.ema_decay)

        torch.testing.assert_close(model_ema['weight'], torch.tensor([1.0]))
        torch.testing.assert_close(fc_ema['weight'], torch.tensor([2.0]))
        model_ema['weight'].add_(5.0)
        torch.testing.assert_close(
            checkpoint['model_state_dict']['weight'], torch.tensor([1.0]))

    def test_warm_start_skips_legacy_alpha_and_classifier(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = nn.Linear(3, 4, bias=False)
                self.alpha = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
                self.log_var = nn.Sequential(nn.BatchNorm1d(4))
                self.fc = nn.Linear(4, 2)

        model = TinyModel()
        source = {
            'trunk.weight': torch.full_like(model.trunk.weight, 2.0),
            'alpha.0.weight': torch.ones(1, 512),
            'alpha.0.bias': torch.ones_like(model.alpha[0].bias),
            'log_var.0.num_batches_tracked': torch.tensor(12),
            'fc.weight': torch.ones_like(model.fc.weight),
        }
        report = load_shape_compatible_state(model, source)

        torch.testing.assert_close(
            model.trunk.weight, torch.full_like(model.trunk.weight, 2.0))
        self.assertEqual(report['loaded'], ['trunk.weight'])
        self.assertIn('alpha.0.weight', report['skipped'])
        self.assertIn('alpha.0.bias', report['skipped'])
        self.assertIn('log_var.0.num_batches_tracked', report['skipped'])
        self.assertIn('fc.weight', report['skipped'])

    def test_checkpoint_container_and_cli_are_v2_scoped(self):
        state = {'weight': torch.tensor([1.0])}
        self.assertIs(checkpoint_state_dict({'state_dict': state}), state)
        args = parse_args(['--raf_path', 'raf-basic'])
        self.assertFalse(hasattr(args, 'backbone'))
        self.assertEqual(args.out_dimension, 64)

    def test_minimum_batch_sampler_merges_only_unsafe_tails(self):
        cases = {
            3: [3],
            5: [5],
            9: [4, 5],
            10: [4, 4, 2],
        }
        for sample_count, expected_sizes in cases.items():
            with self.subTest(sample_count=sample_count):
                sampler = MinimumSizeBatchSampler(
                    list(range(sample_count)), batch_size=4,
                    minimum_batch_size=2)
                batches = list(sampler)
                self.assertEqual([len(batch) for batch in batches], expected_sizes)
                self.assertEqual(len(sampler), len(expected_sizes))
                self.assertEqual(
                    sorted(index for batch in batches for index in batch),
                    list(range(sample_count)))

        with self.assertRaisesRegex(ValueError, 'batch_size'):
            MinimumSizeBatchSampler([0, 1], batch_size=1,
                                    minimum_batch_size=2)
        with self.assertRaisesRegex(ValueError, 'training set'):
            MinimumSizeBatchSampler([0], batch_size=4,
                                    minimum_batch_size=2)


if __name__ == '__main__':
    unittest.main()
