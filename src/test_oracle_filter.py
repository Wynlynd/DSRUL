import tempfile
import unittest
from pathlib import Path

import torch

from train_rafdb import RafDataSet, make_raf_splits


class OracleFilterDataTests(unittest.TestCase):
    def test_split_exposes_only_eligibility_not_clean_class(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            label_directory = root / 'EmoLabel'
            label_directory.mkdir()
            clean_lines = []
            noisy_lines = []
            clean_by_name = {}
            noisy_by_name = {}
            for class_id in range(1, 8):
                for sample_id in range(4):
                    name = 'train_{:02d}_{:02d}.jpg'.format(
                        class_id, sample_id)
                    noisy_label = 2 if class_id == 1 else class_id
                    clean_lines.append('{} {}'.format(name, class_id))
                    noisy_lines.append('{} {}'.format(name, noisy_label))
                    clean_by_name[name] = class_id
                    noisy_by_name[name] = noisy_label
                clean_lines.append(
                    'test_{:02d}.jpg {}'.format(class_id, class_id))

            (label_directory / 'clean.txt').write_text(
                '\n'.join(clean_lines) + '\n', encoding='utf-8')
            (label_directory / 'noisy.txt').write_text(
                '\n'.join(noisy_lines) + '\n', encoding='utf-8')

            train_records, _, _ = make_raf_splits(
                str(root), 'noisy.txt', 'clean.txt', 0.25, 0)

            self.assertIn('prototype_update_eligible', train_records)
            self.assertNotIn('clean_label', train_records)
            for row in train_records.itertuples(index=False):
                self.assertEqual(row.label, noisy_by_name[row.name])
                self.assertEqual(
                    row.prototype_update_eligible,
                    noisy_by_name[row.name] == clean_by_name[row.name])
            self.assertTrue((~train_records['prototype_update_eligible']).any())

            dataset = RafDataSet(str(root), train_records, phase='train')
            self.assertEqual(dataset.prototype_update_eligible.dtype, torch.bool)
            torch.testing.assert_close(
                dataset.prototype_update_eligible,
                torch.tensor(
                    train_records['prototype_update_eligible'].tolist(),
                    dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
