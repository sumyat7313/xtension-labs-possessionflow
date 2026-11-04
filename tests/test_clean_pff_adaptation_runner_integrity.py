"""Independent synthetic checks across campaign, winner refit and operator.

No provider inputs or fitted weights are opened. The end-to-end fixture replaces
data/fitting boundaries with invented observations, while retaining actual file
seals, lifecycle, metric calculation, selection and winner-refit orchestration.
"""
import contextlib
import copy
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import clean_pff_adaptation_campaign as campaign
from tools import clean_pff_adaptation_operator as operator
from tools import clean_pff_adaptation_refit as refit
from tests import test_clean_pff_adaptation_campaign as invented_campaign
from tests.test_clean_pff_adaptation_fit import toy_record


class IndependentMetricAndAdmissionTests(unittest.TestCase):
    def test_scoring_matches_existing_scorer_on_unknowns_and_period_boundaries(self):
        y = np.concatenate((np.repeat(list('HADXH'), [30, 10, 50, 13, 22]),
                            np.repeat(list('ADHX'), [20, 35, 40, 11])))
        prediction = y.copy()
        prediction[3:9] = 'D'
        prediction[125:130] = 'X'
        prediction[175:181] = 'A'
        blocks = [[1, 0, 124], [2, 125, len(y)-1]]
        actual, counts, support = campaign._measure(y, prediction, blocks)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            _, expected = campaign.inputs.prior.score_one(
                Path(temporary), 'synthetic', 999, 'invented', y, prediction, blocks)
        self.assertEqual(actual, expected)
        self.assertEqual(support['known_frames'], int(np.isin(y, list('HAD')).sum()))
        self.assertEqual(counts['short_spells_erased_by_dead'], 0)

    def test_loaded_prediction_cannot_differ_from_authenticated_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'prediction.npy'
            original = np.array(list('HADH'))
            np.save(path, original, allow_pickle=False)
            identity = hashlib.sha256(path.read_bytes()).hexdigest()
            load = np.load
            def change_disk_then_load(source, **kwargs):
                np.save(path, np.array(list('DDDD')), allow_pickle=False)
                return load(source, **kwargs)
            with patch.object(campaign.np, 'load', side_effect=change_disk_then_load):
                try:
                    returned = campaign._array(path, identity)
                except ValueError:
                    return  # Descriptor revalidation is also a correct fail-closed design.
            np.testing.assert_array_equal(returned, original)

    def test_forbidden_validation_matches_fail_before_any_input_io(self):
        forbidden = (*campaign.protocol.INNER_TRAIN_IDS,
                     *campaign.protocol.DEVELOPMENT_IDS, *campaign.protocol.FINAL_IDS)
        with patch.object(campaign.inputs, '_path') as path, \
             patch.object(campaign.inputs, '_json') as read, \
             patch.object(campaign.inputs, '_array') as array:
            for mid in forbidden:
                with self.subTest(mid=mid):
                    for action in (campaign._observations, campaign._reference):
                        with self.assertRaises(ValueError): action(mid, {})
            path.assert_not_called(); read.assert_not_called(); array.assert_not_called()

    def test_unknown_stage_rejected_before_output_or_input_access(self):
        with patch.object(campaign, '_location') as location, patch.object(campaign, '_context') as context:
            for stage in ('predict-A-3814', 'score-3814', 'development', 'final', 'refit-winner', None):
                with self.assertRaises(ValueError): campaign.execute_stage(stage, 'unused')
            location.assert_not_called(); context.assert_not_called()


class IndependentCampaignRefitTests(unittest.TestCase):
    def setUp(self):
        self.fixture = invented_campaign.CampaignTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self._patch(refit, 'ROOT', self.fixture.root)
        self._patch(refit.stages, 'SNAPSHOT_PROTOCOL_SHA256', self.fixture.snapshot['protocol_sha256'])

    def _patch(self, obj, name, value):
        replacement = patch.object(obj, name, value)
        replacement.start(); self.addCleanup(replacement.stop)

    def refit_boundaries(self):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        stack.enter_context(patch.object(refit, '_sources', return_value={'synthetic': 'a'*64}))
        stack.enter_context(patch.object(refit, '_refit_context', return_value={}))
        stack.enter_context(patch.object(refit, '_record', side_effect=lambda mid,ctx: toy_record(mid)))
        stack.enter_context(patch.object(refit.inputs, '_json', side_effect=lambda *a,**kw: copy.deepcopy(self.fixture.parent)))
        return stack

    def test_actual_inner_selection_connects_to_one_refit_and_exact47_support(self):
        self.fixture.advance()
        proof = campaign.verify_completed(self.fixture.out)
        self.assertEqual(proof['selection']['selected_candidate_id'], 'A')
        with self.refit_boundaries():
            result = refit.run_refit(self.fixture.out)
            self.assertEqual(refit.verify_result(self.fixture.out), result)
            self.assertTrue(result['refit_complete'])
            self.assertEqual(result['selected_candidate_id'], 'A')
            self.assertEqual(result['fitting_fixtures'], 47)
            for key in ('promoted', 'submission_eligible', 'final_access_authorized'):
                self.assertIs(result[key], False)
            with patch.object(refit, '_fit') as fit, self.assertRaises(ValueError):
                refit.run_refit(self.fixture.out)
            fit.assert_not_called()

    def test_changed_inner_prediction_prevents_refit_before_fitting_inputs(self):
        self.fixture.advance()
        path = self.fixture.out/f'artifacts/predict-B-{campaign.IDS[-1]}/prediction.npy'
        with path.open('ab') as stream: stream.write(b'changed')
        with self.refit_boundaries(), patch.object(refit, '_refit_context') as context, self.assertRaises(ValueError):
            refit.run_refit(self.fixture.out)
        context.assert_not_called()
        self.assertFalse((self.fixture.root/'runs'/refit.OUT_NAME).exists())

    def test_complete_losing_family_cannot_start_refit(self):
        def predict(item, model):
            self.fixture.prediction_calls.append((model['candidate_id'], item['synthetic_mid']))
            return np.repeat('D', len(self.fixture.y))
        self._patch(campaign, '_predict', predict)
        self.fixture.advance()
        proof = campaign.verify_completed(self.fixture.out)
        self.assertIsNone(proof['selection']['selected_candidate_id'])
        with self.refit_boundaries(), patch.object(refit, '_refit_context') as context, self.assertRaises(ValueError):
            refit.run_refit(self.fixture.out)
        context.assert_not_called()
        self.assertFalse((self.fixture.root/'runs'/refit.OUT_NAME).exists())


class IndependentOperatorConsumptionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve(); (self.root/'runs').mkdir()
        for replacement in (
            patch.object(operator, 'ROOT', self.root),
            patch.object(campaign, 'ROOT', self.root),
            patch.object(operator, 'sources', return_value={'synthetic': 'a'*64}),
            patch.object(operator.stages, 'source_hashes', return_value={'synthetic': 'a'*64}),
            patch.object(operator.stages, 'runtime_versions', return_value={'python': 'synthetic'}),
        ):
            replacement.start(); self.addCleanup(replacement.stop)
        operator.initialize()
        self.out, self.inner = operator._locations()

    def test_zero_exit_without_inner_artifacts_consumes_launch(self):
        with patch.object(operator, '_execute', return_value=0), self.assertRaisesRegex(ValueError, 'not recorded complete'):
            operator.run()
        self.assertTrue(operator.status()['blocked'])
        with patch.object(operator, '_execute') as execute, self.assertRaises(ValueError):
            operator.run()
        execute.assert_not_called()
        self.assertEqual(operator.stages.status(self.inner, self.root)['recorded_completed_stages'], 0)

    def test_worker_without_held_parent_lock_fails_before_manifest_access(self):
        with patch.dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), \
             patch.object(operator, '_manifest') as manifest, self.assertRaisesRegex(ValueError, 'parent operator lock'):
            operator.worker(0, 'a'*32)
        manifest.assert_not_called()

    def test_bad_worker_identity_rejected_before_any_path_access(self):
        with patch.object(operator, '_locations') as locations:
            for index, nonce in ((True, 'a'*32), (-1, 'a'*32), (len(operator.ORDER), 'a'*32), (0, 'wrong')):
                with self.assertRaises(ValueError): operator.worker(index, nonce)
            locations.assert_not_called()


if __name__ == '__main__':
    unittest.main()
