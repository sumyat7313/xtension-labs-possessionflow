"""Independent integrity checks; fixtures contain only synthetic observations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from possession.io import sha256, write_json
from tools import clean_pff_training_diagnosis as diagnosis


class TrainingPartitionIntegrityTests(unittest.TestCase):
    def test_fixed_partition_is_exactly_original_training_forty(self):
        protocol = json.loads((diagnosis.ROOT / 'models/pff-protocol.json').read_text(encoding='utf-8'))
        self.assertEqual(tuple(diagnosis.TRAIN_IDS), tuple(protocol['split']['train']))
        self.assertEqual(len(diagnosis.TRAIN_IDS), 40)
        self.assertFalse(set(diagnosis.TRAIN_IDS) & set(protocol['split']['development']))
        self.assertFalse(set(diagnosis.TRAIN_IDS) & set(protocol['split']['final']))
        self.assertEqual(tuple(diagnosis.NAMES), ('base', 'ha_only', 'full'))

    def test_every_held_out_id_fails_before_observation_io(self):
        protocol = json.loads((diagnosis.ROOT / 'models/pff-protocol.json').read_text(encoding='utf-8'))
        invalid = protocol['split']['development'] + protocol['split']['final'] + [True, False, '3812', 3812., -1]
        with patch.object(diagnosis, 'read', side_effect=AssertionError('Unexpected input read')) as read, \
             patch.object(diagnosis.np, 'load', side_effect=AssertionError('Unexpected array read')) as load:
            for mid in invalid:
                with self.subTest(mid=mid), self.assertRaises(ValueError):
                    diagnosis.observations(mid, {})
            read.assert_not_called(); load.assert_not_called()

    def test_held_out_reference_and_prediction_helpers_guard_before_io(self):
        with patch.object(diagnosis, 'read', side_effect=AssertionError('input read')), \
             patch.object(diagnosis, 'verify_freeze', side_effect=AssertionError('freeze read')), \
             patch.object(diagnosis, 'build_protocol', side_effect=AssertionError('protocol read')):
            for mid in [3813, 3814, True, '3812']:
                for helper in (diagnosis.predict_one, diagnosis.score_one, diagnosis.read_reference):
                    with self.subTest(mid=mid, helper=helper.__name__), self.assertRaises(ValueError):
                        helper(Path('never-read'), mid, {}, {})


class SyntheticInferenceIntegrityTests(unittest.TestCase):
    def test_ha_only_retains_exact_dead_and_conditional_team_evidence(self):
        rng = np.random.default_rng(7)
        full = rng.dirichlet([.3, .3, .3], size=500)
        dead = rng.uniform(0., 1., size=500)
        dead[:2] = [0., 1.-1e-15]
        before = full.copy()
        result = diagnosis.ha_only_probabilities(full, dead)
        np.testing.assert_array_equal(result[:, 2], dead)
        np.testing.assert_allclose(result[:, :2]/result[:, :2].sum(axis=1, keepdims=True),
                                   full[:, :2]/full[:, :2].sum(axis=1, keepdims=True), atol=1e-12, rtol=0)
        np.testing.assert_array_equal(full, before)

    def test_reference_contamination_rejected_before_inference(self):
        with patch.object(diagnosis.robust_control, 'predict_home', side_effect=AssertionError('inference')):
            with self.assertRaises(ValueError):
                diagnosis.ablations({'y': np.array(['H'])}, {}, {})

    def test_only_authenticated_xz_arrays_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            folder = parent / 'cache' / str(diagnosis.TRAIN_IDS[0])
            folder.mkdir(parents=True)
            item = dict(match='pff3812', blocks=[[1, 0, 1]], fps=25, usable=[True],
                        pitch=[105, 68], reference_fields_used_for_inference=False)
            write_json(folder/'item.json', item)
            np.save(folder/'x.npy', np.zeros((2, 98)))
            np.save(folder/'z.npy', np.zeros((2, 58)))
            for name in ('base_home.npy', 'base_dead.npy'):
                (folder/name).write_bytes(b'POISON: old-model inputs must not be read')
            manifest = dict(match_id=3812, files={name: sha256(folder/name) for name in ('item.json', 'x.npy', 'z.npy')})
            write_json(folder/'complete.json', manifest)
            with patch.object(diagnosis, 'PARENT', parent), \
                 patch.object(diagnosis, 'manifests', return_value={'3812': manifest}), \
                 patch.object(diagnosis.np, 'load', wraps=np.load) as load:
                actual = diagnosis.observations(3812, {})
                self.assertEqual([Path(call.args[0]).name for call in load.call_args_list], ['x.npy', 'z.npy'])
                self.assertEqual(set(actual), {'x', 'z', 'blocks', 'fps', 'usable', 'pitch', 'reference_fields_used_for_inference'})
                del actual
                load.reset_mock()
                with (folder/'x.npy').open('ab') as stream:
                    stream.write(b'tampered')
                with self.assertRaisesRegex(ValueError, 'Artifact bytes differ'):
                    diagnosis.observations(3812, {})
                load.assert_not_called()


class DiagnosticAccountingTests(unittest.TestCase):
    def test_matched_short_reference_does_not_become_short_prediction_precision(self):
        # A 1.6-second reference matches a 2.8-second prediction at IoU 4/7.
        reference = np.array(list('H'*40 + 'D'*30))
        prediction = np.array(list('H'*70))
        result = diagnosis.prior.duration_diagnostic(reference, prediction, 25)['bins']
        self.assertEqual(result['1-2s']['n_matched_gt'], 1)
        self.assertEqual(result['1-2s']['n_pred'], 0)
        self.assertEqual(result['1-2s']['n_matched_pred'], 0)
        self.assertIsNone(result['1-2s']['precision'])
        self.assertEqual(result['2-5s']['n_matched_pred'], 1)

    @staticmethod
    def synthetic():
        reference = np.array(list('HADXHADH'))
        labels = {name: np.array(list('DDXDDHAH')) for name in diagnosis.NAMES}
        probabilities = {name: np.tile([.5, .3, .2], (8, 1)) for name in diagnosis.NAMES}
        bins = {name: np.array([0, 1, 0, 1, 255, 1, 0, 255], dtype=np.uint8) for name in diagnosis.BIN_LABELS}
        return reference, labels, probabilities, bins, [[1, 0, 3], [2, 5, 6]]

    def test_unknown_and_inactive_frames_excluded_and_bins_reconcile(self):
        y, labels, probs, bins, blocks = self.synthetic()
        report = diagnosis.aggregate_fixture(y, labels, probs, bins, blocks, 25)
        values = report['models']['base']
        self.assertEqual(values['known_frames'], 5)
        self.assertEqual(values['reference_live'], 3)
        self.assertEqual(values['reference_dead'], 2)
        self.assertEqual(values['false_dead'], 2)
        self.assertEqual(values['missed_dead'], 2)
        self.assertEqual(values['predicted_unknown'], 1)
        self.assertEqual(values['conditional_team_correct'], 1)
        self.assertEqual(values['decoded_team_correct'], 0)
        self.assertEqual(values['decoded_live_coverage'], 1/3)
        for group in report['bins'].values():
            for field in ('known_frames', 'false_dead', 'true_dead', 'missed_dead', 'reference_live', 'reference_dead'):
                self.assertEqual(sum(row['base'][field] for row in group.values()), values[field])

    def test_missing_runs_reset_at_periods_and_do_not_depend_on_reference(self):
        z = np.zeros((60, len(diagnosis.FEATURE_NAMES)))
        item = dict(z=z, fps=25, blocks=[[1, 0, 14], [2, 15, 59]])
        bins = diagnosis.observation_bins(item, np.zeros(60))
        np.testing.assert_array_equal(bins['missing_run_seconds'][:15], np.ones(15))
        np.testing.assert_array_equal(bins['missing_run_seconds'][15:], np.full(45, 2))
        np.testing.assert_array_equal(bins['missing_run_edge_censor'], np.full(60, 4))

    def test_short_spells_respect_unknown_period_and_exact_two_second_barriers(self):
        # Same-team adjacent periods are separate; exact two seconds is excluded.
        y = np.array(list('H'*25 + 'H'*25 + 'X' + 'A'*49 + 'D' + 'H'*50))
        labels = dict(base=np.full(len(y), 'D'), ha_only=np.full(len(y), 'X'), full=np.full(len(y), 'H'))
        report = diagnosis.short_false_dead(y, labels, [[1, 0, 24], [2, 25, len(y)-1]], 25)
        self.assertEqual(report['n_gt'], 3)
        self.assertEqual(report['reference_frames'], 99)
        self.assertEqual(report['models']['base']['spells_erased_by_dead'], 3)
        self.assertEqual(report['models']['ha_only']['spells_erased_by_dead'], 0)
        self.assertEqual(report['models']['full']['spells_erased_by_dead'], 0)
        self.assertEqual(report['models']['base']['false_dead_frames'], 99)

    def test_nonfinite_probabilities_fail_instead_of_publishing(self):
        y, labels, probs, bins, blocks = self.synthetic()
        probs['full'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            diagnosis.aggregate_fixture(y, labels, probs, bins, blocks, 25)


class StageEvidenceIntegrityTests(unittest.TestCase):
    def test_whole_cohort_freeze_rejects_missing_fixture_and_rehashed_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            study = dict(recipe={}, protocol_sha256='fixed', source_hashes={})
            manifests = {str(mid): {'match_id': mid} for mid in diagnosis.TRAIN_IDS}
            files = [f'{name}{suffix}' for name in diagnosis.NAMES for suffix in ('.npy', '-probabilities.npy')] + ['bins.npz']
            for mid in diagnosis.TRAIN_IDS:
                folder = out/'predictions'/str(mid)
                folder.mkdir(parents=True)
                for name in files:
                    (folder/name).write_bytes(b'synthetic opaque prediction')
                proof = dict(match_id=mid, protocol_sha256='fixed', candidate_sha256=diagnosis.CANDIDATE_SHA,
                             config_sha256=diagnosis.CONFIG_SHA,
                             cache_manifest_digest=diagnosis.original.digest(manifests[str(mid)]),
                             files={name: sha256(folder/name) for name in files})
                write_json(folder/'complete.json', proof)
            with patch.object(diagnosis, 'build_protocol', return_value=study), \
                 patch.object(diagnosis, 'manifests', return_value=manifests):
                records = diagnosis.prediction_records(out, study, {})
                self.assertEqual(len(records), 40)
                write_json(out/'predictions-before-references.json', diagnosis.freeze_value(study, records))
                diagnosis.verify_freeze(out, study, {})
                folder = out/'predictions'/str(diagnosis.TRAIN_IDS[-1])
                (folder/'full.npy').write_bytes(b'different synthetic prediction')
                proof = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
                proof['files']['full.npy'] = sha256(folder/'full.npy')
                write_json(folder/'complete.json', proof)
                with self.assertRaisesRegex(ValueError, 'forty-fixture'):
                    diagnosis.verify_freeze(out, study, {})
                (folder/'complete.json').unlink()
                with self.assertRaises(FileNotFoundError):
                    diagnosis.verify_freeze(out, study, {})

    def test_completed_stage_artifact_tamper_rejected_even_with_new_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            study = dict(recipe={}, protocol_sha256='fixed')
            write_json(out/'protocol.json', study)
            proof = dict(stage='initialize', protocol_sha256='fixed', artifacts={'protocol.json': sha256(out/'protocol.json')})
            write_json(out/'stages/initialize.json', proof)
            write_json(out/'state.json', dict(protocol_sha256='fixed', completed={'initialize': sha256(out/'stages/initialize.json')}))
            with patch.object(diagnosis, 'build_protocol', return_value=study):
                diagnosis.authenticate(out, ctx={})
                # A semantically equal JSON rewrite still changes frozen bytes.
                (out/'protocol.json').write_text(json.dumps(study, separators=(',', ':')), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Artifact bytes differ'):
                    diagnosis.authenticate(out, ctx={})

    def test_nonprefix_completed_stages_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            study = dict(recipe={}, protocol_sha256='fixed')
            write_json(out/'protocol.json', study)
            write_json(out/'state.json', dict(protocol_sha256='fixed', completed={'freeze': 'anything'}))
            with patch.object(diagnosis, 'build_protocol', return_value=study), self.assertRaisesRegex(ValueError, 'prefix'):
                diagnosis.authenticate(out, ctx={})

    def test_missing_freeze_blocks_reference_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(diagnosis, 'prediction_records', return_value=[]), \
                 patch.object(diagnosis.prior, 'reference', side_effect=AssertionError('reference accessed')) as ref:
                with self.assertRaises(FileNotFoundError):
                    diagnosis.read_reference(Path(tmp), 3812, dict(protocol_sha256='fixed', source_hashes={}), {})
                ref.assert_not_called()


if __name__ == '__main__':
    unittest.main()
