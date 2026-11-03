import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from possession.io import sha256
from tools import pff_context_prepare as prepare


class ContextPreparationTests(unittest.TestCase):
    def test_held_out_rejected_before_file_access(self):
        protocol = {'split': {'train': [1], 'development': [2], 'final': [3]}}
        for mid in (2, 3, 4, True):
            with self.subTest(mid=mid), self.assertRaisesRegex(ValueError, 'only frozen training'):
                prepare.prepare_match(Path('missing'), Path('missing'), mid, protocol, {})

    def test_authenticated_reuse_and_probability_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'
            output = Path(folder) / 'output'
            cache = source / 'cache' / '1'
            cache.mkdir(parents=True)
            (source / 'candidate.json').write_text('{}')
            (cache / 'complete.json').write_text(json.dumps({'identity': {'source': 'fixed'}}))
            protocol = {'split': {'train': [1], 'development': [2], 'final': [3]}, 'protocol_sha256': 'a' * 64}
            item = {'z': np.zeros((5, 58)), 'fps': 25}
            probabilities = np.tile([.2, .3, .5], (5, 1))
            with mock.patch.object(prepare, 'PARENT_SHA256', sha256(source / 'candidate.json')), \
                 mock.patch.object(prepare.pff_campaign, 'item_at', return_value=item), \
                 mock.patch.object(prepare.pff_training, 'predict_cache', return_value=(None, probabilities)) as predict:
                first = prepare.prepare_match(source, output, 1, protocol, {})
                self.assertEqual(first, prepare.prepare_match(source, output, 1, protocol, {}))
                self.assertEqual(predict.call_count, 1)
                self.assertFalse(first['references_read'])
                np.save(output / '1' / 'parent_probabilities.npy', probabilities[::-1] + .01)
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    prepare.prepare_match(source, output, 1, protocol, {})

    def test_full_rate_probability_validation(self):
        good = np.array([[.1, .2, .7]], dtype=np.float64)
        prepare.validate_probabilities(good, 1)
        for wrong in (good.astype(np.float32), np.array([[.1, .2, .8]]), np.array([[.1, -.2, 1.1]]), np.array([[np.nan, .2, .8]]), np.ones((1, 2))):
            with self.subTest(value=wrong), self.assertRaises(ValueError):
                prepare.validate_probabilities(wrong, 1)


if __name__ == '__main__':
    unittest.main()
