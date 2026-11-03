"""External-only lineage, fixture guards and consumed-stage lifecycle tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from possession.io import write_json, sha256
from tools import external_rebuild as rebuild


class ExternalRebuildTests(unittest.TestCase):
    def test_recipe_rejects_path_injection_and_unknown_parameters(self):
        for key, value in [('profile', 'challenge.json'), ('matches', [3814]),
                           ('baseline', 'models/submission.json'), ('epochs', 0),
                           ('seed', True), ('control_l2', float('nan'))]:
            candidate = rebuild.DEFAULT_RECIPE.copy()
            candidate[key] = value
            with self.assertRaises(ValueError):
                rebuild.recipe(candidate)

    def test_recipe_is_independent_and_bounded(self):
        value = rebuild.recipe()
        value['epochs'] = 12
        self.assertEqual(rebuild.recipe()['epochs'], 8)
        self.assertEqual(rebuild.recipe(value)['epochs'], 12)

    def test_unknown_and_final_fixtures_rejected_before_read(self):
        with patch.object(rebuild, 'read_external') as reader:
            for match in (3814, '3814', 'BAR-ATH', '../3814', 'J03WOH/../../data'):
                with self.assertRaises(ValueError):
                    rebuild.prepare(match, Path('unused'), Path('unused'), None)
            reader.assert_not_called()

    def test_calibrated_regime_rejected_before_cache_read(self):
        with self.assertRaises(ValueError):
            rebuild.cached(Path('unused'), 'J03WOH', 'calibrated')

    def test_folds_exclude_dead_fit_and_whole_held_matches(self):
        scored = []
        for i in range(3):
            train, held = rebuild.split(i)
            self.assertFalse(set(train) & set(held))
            self.assertEqual(set(train) | set(held), set(rebuild.CATALOG))
            self.assertFalse(set(held) & {'J03WOH', 'J03WOY'})
            scored.extend(held)
        self.assertEqual(len(scored), len(set(scored)))
        for value in (True, -1, 3, '0'):
            with self.assertRaises(ValueError):
                rebuild.split(value)

    def test_paths_deny_challenge_and_output_escape(self):
        good = rebuild.ROOT / 'data/external/idsse'
        for data, out in [(rebuild.ROOT / 'data', rebuild.ROOT / 'runs/new'),
                          (good, rebuild.ROOT), (good, rebuild.ROOT / 'runs'),
                          (good, rebuild.ROOT / 'runs/../submission')]:
            with self.assertRaises(ValueError):
                rebuild.paths(data, out)

    def test_pinned_baseline_refuses_modified_bytes_before_load(self):
        with patch.object(rebuild, 'sha256', return_value='f' * 64), patch.object(rebuild, 'load_model') as load:
            with self.assertRaises(ValueError):
                rebuild.base()
            load.assert_not_called()

    def test_link_guard_rejects_before_resolving_path(self):
        with patch.object(Path, 'is_symlink', return_value=True):
            with self.assertRaises(ValueError):
                rebuild.paths(rebuild.ROOT / 'data/external/idsse', rebuild.ROOT / 'runs/new')

    def test_preflight_uses_only_registered_input_kinds(self):
        fake = {'J03WOH': {'name': 'descriptive nonfile metadata',
                          'metadata': (1, 2, 'md5'), 'tracking': (2, 2, 'md5')}}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for kind in ('metadata', 'tracking'):
                (root / f'J03WOH_{kind}.xml').write_bytes(b'ok')
            with patch.object(rebuild, 'CATALOG', fake), \
                 patch.object(rebuild, 'paths', return_value=(root, root / 'out')), \
                 patch.object(rebuild, 'plan', return_value={}), \
                 patch.object(rebuild, 'digest', return_value='md5'):
                result = rebuild.preflight(root, root / 'out')
            self.assertEqual(len(result['inputs']), 2)

    def test_base_strips_old_learned_control(self):
        base, _ = rebuild.base()
        self.assertNotIn('control_model', base)
        self.assertNotIn('robust_control_model', base)
        self.assertNotIn('temporal_model', base)
        self.assertNotIn('joint_model', base)
        self.assertEqual(base['metadata']['dead_training_matches'], ['J03WOH', 'J03WOY'])

    def test_weights_never_cross_period_or_unknown_barriers(self):
        y = np.array(list('XHHADXXAAHDX'))
        weights = rebuild.joint_weights(y, [(1, 1, 4), (2, 7, 10)], 2.)
        self.assertTrue((weights[[0, 5, 6, 11]] == 0).all())
        self.assertTrue((weights[[1, 2, 3, 4, 7, 8, 9, 10]] >= 2).all())

    def test_seal_detects_changed_and_extra_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'one').write_text('original')
            rebuild._seal(folder)
            rebuild._verify_seal(folder)
            (folder / 'extra').write_text('untracked')
            with self.assertRaises(ValueError):
                rebuild._verify_seal(folder)

    def test_independent_policy_accepts_complete_external_dag(self):
        frozen = {'source_hashes': {'tools/synthetic.py': 'a' * 64},
                  'inputs': {'J03WOH_tracking.xml': {'sha256': 'b' * 64}}}
        value = rebuild.lineage(frozen, 'c' * 64, 'external-joint')
        self.assertEqual(value['candidate_sha256'], 'c' * 64)
        self.assertIn('b' * 64, value['fit_inputs'])
        candidate = next(v for v in value['artifacts'] if v['sha256'] == 'c' * 64)
        self.assertEqual(set(candidate['parents']), {rebuild.BASE_SHA, rebuild.CONFIG_SHA, 'a' * 64, 'b' * 64})

    def test_independent_policy_rejects_contaminated_ancestor(self):
        frozen = {'source_hashes': {'tools/synthetic.py': 'a' * 64},
                  'inputs': {'BAR-ATH.xml': {'sha256': 'b' * 64}}}
        with self.assertRaises(ValueError):
            rebuild.lineage(frozen, 'c' * 64, 'external-joint')

    def records(self, gain=.02):
        rows = []
        for held in rebuild.FOLDS:
            for match in held:
                for regime in rebuild.REGIMES:
                    for i, name in enumerate(rebuild.NAMES):
                        value = .5 + i * gain
                        rows.append(dict(match=match, regime=regime, candidate=name,
                            spellF1=value, macroF1=value, dead_precision=value, dead_recall=value,
                            control_diagnostics={'transitions': {'recall': value},
                            'spell_recall_by_duration': {'<2s': {'recall': value}}}))
        return rows

    def test_complete_gates_choose_joint_only_on_all_safeguards(self):
        self.assertEqual(rebuild.select(self.records())['selected'], 'external-joint')
        self.assertEqual(rebuild.select(self.records(0))['selected'], 'external-control')
        rows = self.records()
        rows[1]['dead_recall'] = .1
        self.assertFalse(rebuild.select(rows)['development_gate_passed'])

    def test_selection_rejects_missing_duplicate_and_nonfinite(self):
        rows = self.records()
        invalid = copy.deepcopy(rows)
        invalid[0]['spellF1'] = float('nan')
        for candidate in (rows[:-1], rows + [rows[0]], invalid):
            with self.assertRaises(ValueError):
                rebuild.select(candidate)

    def _stage_fixture(self, folder):
        protocol = dict(stages=['prepare-J03WOH', 'prepare-J03WOY'], source_hashes={'synthetic': 'x'})
        write_json(folder / 'protocol.json', protocol)
        return protocol, dict(protocol_sha256=sha256(folder / 'protocol.json'), completed={})

    def test_successful_stage_is_checkpointed_without_promotion(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            protocol, state = self._stage_fixture(folder)
            def prepare(match, data, output, config):
                output.mkdir(parents=True)
                (output / 'synthetic').write_text('no raw data')
                rebuild._seal(output)
            with patch.object(rebuild, 'paths', return_value=(folder, folder)), \
                 patch.object(rebuild, '_authenticate', return_value=(protocol, state)), \
                 patch.object(rebuild, 'base', return_value=({}, None)), \
                 patch.object(rebuild, 'source_hashes', return_value=protocol['source_hashes']), \
                 patch.object(rebuild, 'prepare', side_effect=prepare):
                result = rebuild.run(folder, folder)
            self.assertEqual(result['status'], 'in_progress')
            self.assertEqual(result['next_stage'], 'prepare-J03WOY')
            self.assertFalse(result['promoted'])
            self.assertFalse(result['final_accessed'])
            self.assertTrue((folder / 'prepare-J03WOH-started.json').exists())
            self.assertIn('prepare-J03WOH', json.loads((folder / 'state.json').read_text())['completed'])

    def test_failed_stage_is_consumed_and_never_silently_retried(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            protocol, state = self._stage_fixture(folder)
            with patch.object(rebuild, 'paths', return_value=(folder, folder)), \
                 patch.object(rebuild, '_authenticate', return_value=(protocol, state)), \
                 patch.object(rebuild, 'base', return_value=({}, None)), \
                 patch.object(rebuild, 'prepare', side_effect=RuntimeError('failure')) as prepare:
                with self.assertRaises(RuntimeError):
                    rebuild.run(folder, folder)
                with self.assertRaises(FileExistsError):
                    rebuild.run(folder, folder)
                self.assertEqual(prepare.call_count, 1)


if __name__ == '__main__':
    unittest.main()
