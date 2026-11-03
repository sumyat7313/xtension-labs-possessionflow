"""Synthetic-only checks of the fixed clean-model development evaluator."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from possession.io import sha256, write_json
from tools import external_pff_development as study


class ExternalPFFDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def item(self):
        return dict(x=np.zeros((8, len(study.robust_control.FEATURE_NAMES))),
                    z=np.zeros((8, 58)), blocks=[[1, 1, 3], [2, 5, 7]], fps=25.,
                    usable=[True, False], pitch=[105., 68.], reference_fields_used_for_inference=False)

    def model(self):
        return {'feature_names': ['synthetic'] * 58, 'robust_control_model': {'clean': True},
                'joint_model': {'clean': True}}

    def test_recipe_is_exact_fixed_candidate(self):
        self.assertEqual(study.recipe()['candidate_sha256'], study.CANDIDATE_SHA)
        for value in ({}, {'schema_version': True, 'candidate_sha256': study.CANDIDATE_SHA},
                      dict(study.recipe(), split='final'), dict(study.recipe(), penalty=.1),
                      dict(study.recipe(), candidate_sha256=study.original.BASELINE_SHA256)):
            with self.assertRaises(ValueError):
                study.recipe(value)

    def test_final_unknown_and_noninteger_ids_rejected_before_io(self):
        with patch.object(study, 'read') as read, patch.object(study, 'checked_hash') as check:
            for mid in (3814, 3815, 3824, 3830, 3837, 3840, 3841, 3848, 3852,
                        10506, 10510, 10511, '3813', True, 3813., 999):
                with self.assertRaises(ValueError):
                    study.observations(mid, {})
                with self.assertRaises(ValueError):
                    study.predict_one(self.root, mid, {}, {})
                with self.assertRaises(ValueError):
                    study.read_reference(self.root, mid, {}, {})
            read.assert_not_called(); check.assert_not_called()

    def test_observation_loader_never_opens_old_upstream_probabilities(self):
        folder = self.root / 'cache/3813'
        folder.mkdir(parents=True)
        item = self.item()
        for name in ('x', 'z'):
            np.save(folder / f'{name}.npy', item[name], allow_pickle=False)
        for name in ('base_home', 'base_dead'):
            (folder / f'{name}.npy').write_bytes(b'poison; must never be read')
        manifest = {'3813': {'files': {n + '.npy': sha256(folder / (n + '.npy')) for n in ('x', 'z')}}}
        meta = {k: v for k, v in item.items() if k not in ('x', 'z')}
        real_load = np.load
        with patch.object(study, 'PARENT', self.root), \
             patch.object(study, '_observation_meta', return_value=meta), \
             patch.object(study, 'manifests', return_value=manifest), \
             patch.object(study.np, 'load', wraps=real_load) as load:
            loaded = study.observations(3813, {})
        self.assertEqual({Path(call.args[0]).name for call in load.call_args_list}, {'x.npy', 'z.npy'})
        self.assertEqual(set(loaded), set(item))
        loaded['x']._mmap.close()
        loaded['z']._mmap.close()

    def test_candidate_recomputes_both_heads_and_keeps_period_gaps_unknown(self):
        item, model = self.item(), self.model()
        home = np.linspace(.2, .8, 8)
        dead = np.linspace(.1, .3, 8)
        probabilities = np.tile([.7, .2, .1], (8, 1))
        with patch.object(study.robust_control, 'predict_home', return_value=home) as home_fn, \
             patch.object(study, 'predict_dead', return_value=dead) as dead_fn, \
             patch.object(study, 'joint_probabilities', return_value=probabilities) as joint, \
             patch.object(study, 'decode', return_value=np.zeros(3, dtype=int)):
            labels = study.predict_observations(item, model, None)
        self.assertEqual(''.join(labels), 'XHHHXXXX')
        self.assertIs(home_fn.call_args.args[0], model['robust_control_model'])
        self.assertIs(dead_fn.call_args.args[1], model)
        np.testing.assert_array_equal(joint.call_args.args[0], dead)
        np.testing.assert_array_equal(joint.call_args.args[1], home)

    def test_probability_or_reference_injection_rejected_before_inference(self):
        for key in ('base_home', 'base_dead', 'y', 'events'):
            item = self.item(); item[key] = np.ones(8)
            with patch.object(study.robust_control, 'predict_home') as predictor:
                with self.assertRaises(ValueError):
                    study.predict_observations(item, self.model(), None)
                predictor.assert_not_called()

    def test_invalid_features_and_unexpected_learned_heads_fail(self):
        candidates = []
        for key, value in [('fps', 24), ('usable', [True]), ('reference_fields_used_for_inference', True),
                           ('x', np.zeros((8, 2))), ('z', np.full((8, 58), np.nan))]:
            item = self.item(); item[key] = value; candidates.append((item, self.model()))
        model = self.model(); model['temporal_model'] = {}
        candidates.append((self.item(), model))
        for item, model in candidates:
            with self.assertRaises(ValueError):
                study.predict_observations(item, model, None)

    def test_cache_file_tampering_is_detected(self):
        path = self.root / 'artifact.json'; path.write_text('original')
        expected = sha256(path); path.write_text('changed')
        with self.assertRaises(ValueError):
            study.checked_hash(path, expected)

    def test_wrong_source_protocol_prevents_prediction_reads(self):
        with patch.object(study, 'build_protocol', return_value={'new': True}), \
             patch.object(study, 'observations') as observations:
            with self.assertRaises(ValueError):
                study.predict_one(self.root, 3813, {'recipe': study.recipe()}, {})
            observations.assert_not_called()

    def test_direct_prediction_helper_refuses_consumed_references(self):
        write_json(self.root / 'reference-access.json', {'consumed': True})
        with patch.object(study, 'observations') as observations, patch.object(study, 'build_protocol') as build:
            with self.assertRaisesRegex(ValueError, 'regenerated'):
                study.predict_one(self.root, 3813, {}, {})
            observations.assert_not_called(); build.assert_not_called()

    def test_direct_prediction_helper_refuses_frozen_cohort(self):
        write_json(self.root / 'predictions-before-references.json', {'frozen': True})
        with patch.object(study, 'observations') as observations:
            with self.assertRaisesRegex(ValueError, 'cohort freeze'):
                study.predict_one(self.root, 3813, {}, {})
            observations.assert_not_called()

    def test_incomplete_cohort_prevents_first_reference_access(self):
        with patch.object(study, 'prediction_records', side_effect=FileNotFoundError('missing fixture')), \
             patch.object(study, 'begin_access') as access, patch.object(study, 'read_reference') as refs:
            with self.assertRaises(FileNotFoundError):
                study.score(self.root, {}, {})
            access.assert_not_called(); refs.assert_not_called()

    def test_access_marker_is_exclusive_and_consumes_partial_scoring(self):
        value = {'references_consumed': True, 'candidate_sha256': study.CANDIDATE_SHA}
        with patch.object(study, 'access_value', return_value=value):
            study.begin_access(self.root, {})
            with self.assertRaises(FileExistsError):
                study.begin_access(self.root, {})
        self.assertEqual(json.loads((self.root / 'reference-access.json').read_text()), value)

    def test_reference_requires_matching_access_marker(self):
        write_json(self.root / 'reference-access.json', {'bad': True})
        with patch.object(study, 'verify_freeze'), patch.object(study, 'access_value', return_value={'good': True}), \
             patch.object(study.prior, 'reference') as reference:
            with self.assertRaises(ValueError):
                study.read_reference(self.root, 3813, {}, {})
            reference.assert_not_called()

    def test_reference_change_fails_before_reconstruction(self):
        write_json(self.root / 'reference-access.json', {})
        with patch.object(study, 'verify_freeze'), patch.object(study, 'access_value', return_value={}), \
             patch.object(study, 'manifests', return_value={'3813': {'files': {'alignment.json': 'a'}, 'identity': {'events': 'b'}}}), \
             patch.object(study, 'checked_hash', side_effect=ValueError('changed')), \
             patch.object(study.prior, 'reference') as reference:
            with self.assertRaises(ValueError):
                study.read_reference(self.root, 3813, {}, {})
            reference.assert_not_called()

    def test_historical_control_metrics_must_match_before_gate(self):
        folder = self.root / 'predictions/3813'; folder.mkdir(parents=True)
        for name in study.NAMES:
            np.save(folder / f'{name}.npy', np.array(list('HAD')))
        detail = {'temporal_diagnostics': {'dead_segments': {'n_gt': 1}, 'short_possessions': {'n_gt': 1}}}
        with patch.object(study, 'IDS', (3813,)), patch.object(study, 'verify_freeze'), \
             patch.object(study, 'begin_access'), patch.object(study, '_observation_meta', return_value={'blocks': [[1, 0, 2]]}), \
             patch.object(study, 'read_reference', return_value=(np.array(list('HAD')), {})), \
             patch.object(study.prior, 'score_one', return_value=(detail, {'spell': .1})), \
             patch.object(study.precision, 'development_gate') as gate:
            with self.assertRaisesRegex(ValueError, 'Historical control'):
                study.score(self.root, {}, {})
            gate.assert_not_called()

    def _synthetic_protocol(self):
        value = dict(schema_version=1, recipe=study.recipe(), stages=list(study.STAGES),
                     source_hashes={'synthetic': 'a'}, candidate_sha256=study.CANDIDATE_SHA)
        value['protocol_sha256'] = study.original.digest(value)
        return value

    def test_initialize_is_only_one_bounded_stage_and_does_not_predict(self):
        out = self.root / 'output'
        protocol = self._synthetic_protocol()
        with patch.object(study, 'location', return_value=out), patch.object(study, 'context', return_value={}), \
             patch.object(study, 'build_protocol', return_value=protocol), patch.object(study, 'predict_one') as predict:
            summary = study.run(out)
        self.assertEqual(summary['stage_completed'], 'initialize')
        self.assertEqual(summary['next_stage'], 'predict-3813')
        self.assertEqual(summary['status'], 'in_progress')
        self.assertEqual(summary['protocol_file_sha256'], sha256(out / 'protocol.json'))
        self.assertEqual(summary['protocol_sha256'], protocol['protocol_sha256'])
        self.assertFalse(summary['final_access_authorized'])
        predict.assert_not_called()

    def test_failed_prediction_stage_cannot_be_retried(self):
        out = self.root / 'output'
        protocol = self._synthetic_protocol()
        with patch.object(study, 'location', return_value=out), patch.object(study, 'context', return_value={}), \
             patch.object(study, 'build_protocol', return_value=protocol):
            study.run(out)
            with patch.object(study, 'predict_one', side_effect=RuntimeError('interrupted')) as predict:
                with self.assertRaises(RuntimeError):
                    study.run(out)
                with self.assertRaises(FileExistsError):
                    study.run(out)
                self.assertEqual(predict.call_count, 1)

    def test_source_drift_after_stage_cannot_commit_completion(self):
        out = self.root / 'output'
        protocol = self._synthetic_protocol()
        with patch.object(study, 'location', return_value=out), patch.object(study, 'context', return_value={}), \
             patch.object(study, 'build_protocol', side_effect=[protocol, dict(protocol, candidate_sha256='changed')]):
            with self.assertRaises(ValueError):
                study.run(out)
        self.assertTrue((out / 'initialize-started.json').exists())
        self.assertFalse((out / 'state.json').exists())

    def test_authentication_rejects_skipped_stage(self):
        protocol = self._synthetic_protocol()
        write_json(self.root / 'protocol.json', protocol)
        write_json(self.root / 'state.json', {'protocol_sha256': protocol['protocol_sha256'], 'completed': {'score': 'a'}})
        with patch.object(study, 'build_protocol', return_value=protocol):
            with self.assertRaisesRegex(ValueError, 'frozen order'):
                study.authenticate(self.root, ctx={})

    def test_locations_cannot_overlap_historical_or_production(self):
        for path in (study.ROOT / 'submission', study.PARENT, study.REBUILD / 'inside', study.REBUILD.parent):
            with self.assertRaises(ValueError):
                study.location(path)

    def test_fifteen_stages_have_no_final_operation(self):
        self.assertEqual(len(study.STAGES), 15)
        self.assertFalse(any('final' in stage for stage in study.STAGES))
        self.assertEqual(set(study.CACHE_HASHES), {str(m) for m in study.IDS})
        self.assertEqual(study.original.digest(study.CACHE_HASHES), study.CACHE_MAP_DIGEST)


if __name__ == '__main__':
    unittest.main()
